# SPDX-License-Identifier: GPL-3.0-only
"""Small real MOSS class: arithmetic compilation, masks, LoRA swaps and restoration."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
import types
from _paths import configure
paths=configure(Path(__file__).stem)
sys.path.insert(0,str(paths.comfy));sys.argv=[sys.argv[0]]
import torch
import comfy.ops,comfy.sd
from comfy.model_patcher import ModelPatcher
from torch._dynamo.utils import counters

def load(name,path,package=False):
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)] if package else None)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

upstream=load('_moss_native_validation',paths.moss/'model.py')
plugin=load('_moss_precompile_validation',paths.root/'__init__.py',True)
runtime=sys.modules[plugin.__name__+'.runtime'];hot=sys.modules[plugin.__name__+'.hot_lora'];full=sys.modules[plugin.__name__+'.full_precompile']
torch.set_grad_enabled(False);torch.manual_seed(5132)
diffusion=upstream.MOSSImageTransformer(dim=64,ffn_dim=128,num_heads=4,num_layers=2,
    text_dim=32,freq_dim=16,latent_channels=8,device='cuda',dtype=torch.float32,operations=comfy.ops.manual_cast)
for parameter in diffusion.parameters():parameter.uniform_(-.1,.1)
base=torch.nn.Module();base.diffusion_model=diffusion;base.model_config=types.SimpleNamespace(unet_config={})
model=ModelPatcher(base,torch.device('cuda'),torch.device('cpu'))
compiled=runtime.patch_moss(model);originals=[block.forward for block in diffusion.layers]
results={'model':'MOSSImageTransformer','quantized':False,'checkpoint_loaded':False,'cases':[]}

def predict(patcher,x,t,context,mask):
    patcher.patch_model(load_weights=False)
    try:return diffusion(x,t,context,attention_mask=mask).detach().clone()
    finally:
        patcher.unpatch_model(unpatch_weights=False)
        assert all(layer.forward==original for layer,original in zip(diffusion.layers,originals))

for batch,height,width,tokens in [(1,3,4,3),(2,2,3,5)]:
    x=torch.randn(batch,8,height,width,device='cuda');t=torch.full((batch,),.5,device='cuda')
    context=torch.randn(batch,tokens,32,device='cuda');mask=torch.ones(batch,tokens,device='cuda',dtype=torch.bool)
    if batch==2:mask[0,3:]=False
    expected=predict(model,x,t,context,mask);actual=predict(compiled,x,t,context,mask);torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
    record={'batch':batch,'latent':[height,width],'text_tokens':tokens,'max_abs':float((actual-expected).abs().max())}
    # Changing masked-away token values must not change either route.
    altered=context.clone();altered[~mask]=99
    again=predict(compiled,x,t,altered,mask);torch.cuda.synchronize()
    torch.testing.assert_close(again,actual,rtol=0,atol=0)
    record['masked_text_isolated']=True;results['cases'].append(record)

# Synthetic rank4/rank8 adapters target a real MOSS projection. Keep the base
# matrix fixed, compare with ComfyUI's own bypass reference, then switch back.
x=torch.randn(1,8,3,4,device='cuda');t=torch.tensor([.5],device='cuda');context=torch.randn(1,3,32,device='cuda');mask=torch.ones(1,3,device='cuda',dtype=torch.bool)
key='lora_unet_layers_0_proj_q';projection=diffusion.layers[0].proj['q'];native_projection=projection.forward;weight=projection.weight.clone()
def adapter(rank):return {key+'.lora_down.weight':torch.randn(rank,64)*.03,key+'.lora_up.weight':torch.randn(64,rank)*.03,key+'.alpha':torch.tensor(float(rank))}
a,b=adapter(4),adapter(8)
enabled=hot.enable_hot_lora(compiled)
A,_=comfy.sd.load_lora_for_models(enabled,None,a,.7,0)
B,_=comfy.sd.load_lora_for_models(enabled,None,b,-.3,0)
stack,_=comfy.sd.load_lora_for_models(A,None,b,-.3,0)
RA,_=comfy.sd.load_bypass_lora_for_models(model,None,a,.7,0)
RB,_=comfy.sd.load_bypass_lora_for_models(model,None,b,-.3,0)
ref_a=predict(RA,x,t,context,mask);ref_b=predict(RB,x,t,context,mask)
out_a=predict(A,x,t,context,mask);before=counters['stats']['unique_graphs'];out_b=predict(B,x,t,context,mask);returned=predict(A,x,t,context,mask)
torch.testing.assert_close(out_a,ref_a,rtol=1e-4,atol=1e-5);torch.testing.assert_close(out_b,ref_b,rtol=1e-4,atol=1e-5)
torch.testing.assert_close(returned,out_a,rtol=0,atol=0);assert not torch.equal(out_a,out_b)
results['hot_lora']={'rank4_max_abs':float((out_a-ref_a).abs().max()),'rank8_max_abs':float((out_b-ref_b).abs().max()),'return_exact':True,'extra_arithmetic_graphs_on_swap':counters['stats']['unique_graphs']-before}
# Use direct additive projection oracle for stacks, avoiding assumptions about
# the upstream reference injection-group ejection order.
projection_input=torch.randn(1,7,64,device='cuda');base_out=torch.nn.functional.linear(projection_input,projection.weight,projection.bias)
expected_stack=base_out
for raw,factor in [(a,.7),(b,-.3)]:expected_stack=expected_stack+torch.nn.functional.linear(torch.nn.functional.linear(projection_input,raw[key+'.lora_down.weight'].cuda()),raw[key+'.lora_up.weight'].cuda())*factor
stack.patch_model(load_weights=False)
try:
    actual_stack=projection(projection_input);torch.testing.assert_close(actual_stack,expected_stack,rtol=0,atol=0)
finally:stack.unpatch_model(unpatch_weights=False)
assert projection.forward==native_projection and torch.equal(weight,projection.weight)
results['hot_lora']['stack_projection_exact']=True;results['base_weights_unchanged']=True

# Full mode must report upstream data-dependent mask logic, not modify it.
try:
    full.FullGraphCall(diffusion.forward,'MOSS DiT')(x,t,context,attention_mask=mask)
    torch.cuda.synchronize();results['strict_full_graph_supported']=True
except full.FullGraphError as error:
    results['strict_full_graph_supported']=False;results['strict_full_error']=str(error)
results['restored']=all(layer.forward==original for layer,original in zip(diffusion.layers,originals))
results['passed']=True
(paths.out/'result.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2),flush=True)
del model,compiled,enabled,A,B,stack,RA,RB,base,diffusion
gc.collect()
