# SPDX-License-Identifier: GPL-3.0-only
"""Architecture identification only; no dependency on checkpoint/quant layouts."""

def is_moss(model):
    diffusion=model.get_model_object("diffusion_model")
    return type(diffusion).__name__=="MOSSImageTransformer" and all(
        hasattr(diffusion,name) for name in (
            "layers","video_embedder","text_embedder","shared_block_modulation",
            "final_norm_video","final_linear_video","video_head_modulation",
        )
    )
