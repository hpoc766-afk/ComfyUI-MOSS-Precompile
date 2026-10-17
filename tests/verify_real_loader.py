# SPDX-License-Identifier: GPL-3.0-only
"""Actual official MOSS loading and one native DiT forward; compilation stays off."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
from _paths import configure
paths=configure(Path(__file__).stem)
sys.path.insert(0,str(paths.comfy));sys.argv=[sys.argv[0]]
import torch
import comfy.model_management as mm
from utils.extra_config import load_extra_path_config
extra=paths.comfy/'extra_model_paths.yaml'
if extra.is_file():load_extra_path_config(str(extra))
spec=importlib.util.spec_from_file_location('_moss_real_loader_check',paths.root/'__init__.py',submodule_search_locations=[str(paths.root)])
plugin=importlib.util.module_from_spec(spec);sys.modules[spec.name]=plugin;spec.loader.exec_module(plugin)
torch.set_grad_enabled(False)
model,clip,vae=plugin.MOSSPrecompileLoader().load(paths.args.dit_name,paths.args.text_encoder_name,paths.args.vae_name)
diffusion=model.get_model_object('diffusion_model')
assert plugin.is_moss(model)
assert not model.object_patches and not model.get_attachment(sys.modules[spec.name+'.hot_lora']._ENABLED)
mm.load_models_gpu([model],memory_required=256*1024*1024)
device=mm.get_torch_device();torch.manual_seed(731)
x=torch.randn(1,diffusion.latent_channels,4,4,device=device,dtype=diffusion.dtype)
context=torch.randn(1,4,4096,device=device,dtype=diffusion.dtype)
time=torch.tensor([.5],device=device)
output=diffusion(x,time,context);torch.cuda.synchronize()
assert output.shape==x.shape and torch.isfinite(output).all()
result={'passed':True,'model_class':type(diffusion).__name__,'clip_class':type(clip.cond_stage_model).__name__,
 'vae_class':type(vae.first_stage_model).__name__,'output_shape':list(output.shape),
 'native_forward_finite':True,'compile_enabled':False,'model_object_patches':len(model.object_patches),
 'scope':'Native model/TE/VAE loading plus one small DiT forward; no TE encode, VAE encode/decode, sampling or full-image benchmark.'}
(paths.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
mm.unload_all_models();del model,clip,vae,diffusion;gc.collect()
