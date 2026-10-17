# SPDX-License-Identifier: GPL-3.0-only
import folder_paths
import nodes

from .backend import get_backend
from .full_precompile import patch_full_models
from .hot_lora import enable_hot_lora
from .model_kind import is_moss
from .runtime import patch_moss


DTYPES=("default","bf16","fp32","fp8_e4m3fn","fp8_e4m3fn_fast","fp8_e5m2")
OPTIONS={
    "precompile":("BOOLEAN",{"default":False,"tooltip":"编译MOSS主干外围的加法/乘法，保留原Attention与MLP。"}),
    "hot_lora":("BOOLEAN",{"default":False,"tooltip":"后续普通LoRA节点使用兼容的低秩旁路；保留真实rank，不重新量化底座。"}),
    "full_precompile":("BOOLEAN",{"default":False,"tooltip":"分别尝试TE Transformer、DiT、VAE编解码严格完整图，优先于逐元素模式。不修改原生mask或量化后端，不支持时报错。"}),
}


def prepare_models(model,clip,vae,precompile=False,hot_lora=False,full_precompile=False):
    if not is_moss(model):raise ValueError("输入MODEL不是官方MOSS-Image模型。")
    if full_precompile:
        model,clip,vae=patch_full_models(model,clip,vae)
    elif precompile:
        model=patch_moss(model)
    if hot_lora:model=enable_hot_lora(model)
    return model,clip,vae


class MOSSPrecompileLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "dit_name":(folder_paths.get_filename_list("diffusion_models"),),
            "text_encoder_name":(folder_paths.get_filename_list("text_encoders"),),
            "vae_name":(nodes.VAELoader.vae_list(nodes.VAELoader),),
            "weight_dtype":(list(DTYPES),{"default":"default"}),
            "text_device":(["default","cpu"],{"default":"default"}),
        },"optional":dict(OPTIONS)}
    RETURN_TYPES=("MODEL","CLIP","VAE")
    RETURN_NAMES=("模型","文本编码器","变分自编码器")
    FUNCTION="load";CATEGORY="MOSS/模型加载";EXPERIMENTAL=True
    DESCRIPTION="使用官方MOSS后端加载主干、Qwen3.5文本编码器和VAE，可选预编译与LoRA热切换。"

    def load(self,dit_name,text_encoder_name,vae_name,weight_dtype="default",text_device="default",
             precompile=False,hot_lora=False,full_precompile=False):
        backend=get_backend()
        if weight_dtype not in backend.DTYPES:raise ValueError("所选精度不受当前MOSS后端支持。")
        model=backend.load_model(folder_paths.get_full_path_or_raise("diffusion_models",dit_name),weight_dtype)
        clip=backend.load_clip(folder_paths.get_full_path_or_raise("text_encoders",text_encoder_name),text_device)
        vae,=nodes.VAELoader().load_vae(vae_name)
        return prepare_models(model,clip,vae,precompile,hot_lora,full_precompile)


class MOSSPrecompile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"model":("MODEL",),"clip":("CLIP",),"vae":("VAE",)},"optional":dict(OPTIONS)}
    RETURN_TYPES=("MODEL","CLIP","VAE")
    RETURN_NAMES=("模型","文本编码器","变分自编码器")
    FUNCTION="apply";CATEGORY="MOSS/模型加载";EXPERIMENTAL=True
    DESCRIPTION="连接已有MOSS加载器输出，保持原权重格式及计算后端，选择预编译与LoRA热切换。"

    def apply(self,model,clip,vae,precompile=False,hot_lora=False,full_precompile=False):
        return prepare_models(model,clip,vae,precompile,hot_lora,full_precompile)


NODE_CLASS_MAPPINGS={"MOSSPrecompileLoader":MOSSPrecompileLoader,"MOSSPrecompile":MOSSPrecompile}
NODE_DISPLAY_NAME_MAPPINGS={"MOSSPrecompileLoader":"MOSS 模型加载器","MOSSPrecompile":"MOSS 模型预编译"}
