# SPDX-License-Identifier: GPL-3.0-only
"""MOSS block arithmetic adapter; attention, MLP and weight backend stay native."""
import torch
from .model_kind import is_moss


def _add(a,b):return a+b

def _multiply(a,b):return a*b


def compiled_block(block,original,add,multiply):
    source=getattr(original,"__func__",original)
    helpers=getattr(source,"__globals__",{})
    attention=helpers.get("_attention");situ=helpers.get("_situ")
    if not callable(attention) or not callable(situ):
        raise ValueError("MOSS block接口不兼容：缺少原生_attention或_situ函数。")

    def forward(x,rope,modulation):
        # Preserve the upstream per-operation order and low-precision outputs.
        # No token padding, quantization inspection or replacement GEMMs.
        shift_a,scale_a,gate_a,shift_f,scale_f,gate_f=modulation.unbind(1)
        y=add(multiply(block.norm1(x),add(1,scale_a[:,None])),shift_a[:,None])
        dtype=block.proj["q"].weight.dtype
        y=y.to(dtype)
        q=block.q_norm(block.proj["q"](y).unflatten(-1,(block.heads,-1))).flatten(2)
        k=block.k_norm(block.proj["k"](y).unflatten(-1,(block.heads,-1))).flatten(2)
        attended=attention(q,k,block.proj["v"](y),rope,block.heads)
        x=add(x,multiply(block.proj["o"](attended),gate_a[:,None]))
        y=add(multiply(block.norm2(x),add(1,scale_f[:,None])),shift_f[:,None])
        hidden,gate=block.proj["ffn_up_gate"](y.to(dtype)).chunk(2,-1)
        return add(x,multiply(block.proj["ffn_down"](situ(hidden,gate)),gate_f[:,None]))
    return forward


def patch_moss(model):
    if not is_moss(model):raise ValueError("请选择官方MOSS-Image加载器输出的模型。")
    diffusion=model.get_model_object("diffusion_model")
    # Each add/multiply returns a typed tensor before the next operation. Avoid
    # fusing an entire expression across the original BF16 rounding boundaries.
    settings={"backend":"inductor","fullgraph":True,"dynamic":True,
              "options":{"triton.cudagraphs":False}}
    add=torch.compile(_add,**settings);multiply=torch.compile(_multiply,**settings)
    patched=model.clone()
    for index,block in enumerate(diffusion.layers):
        key=f"diffusion_model.layers.{index}.forward"
        original=model.get_model_object(key)
        patched.add_object_patch(key,compiled_block(block,original,add,multiply))
    return patched
