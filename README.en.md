# ComfyUI-MOSS-Precompile

Provides model loading, optional element-by-element precompilation, and LoRA hot-switching for MOSS-Image. The weight parsing and calculation backend uses the official MOSS node, and the compilation layer does not rely on a specific quantization format, does not expand weights, and does not register format-specific operators.

## Install

First press [MOSS-Image Official Instructions](https://github.com/OpenMOSS/MOSS-Image/tree/main/comfyui) to install the 'moss-image' backend, then execute it in the ComfyUI directory:

`bash
git clone https://github.com/hpoc766-afk/ComfyUI-MOSS-Precompile.git custom_nodes/ComfyUI-MOSS-Precompile
`

Use after restarting ComfyUI. Update:

`bash
git -C custom_nodes/ComfyUI-MOSS-Precompile pull --ff-only
`

Use existing and compatible ComfyUI, PyTorch, and model backends. Enabling precompilation also requires an available Inductor compilation environment; this node does not automatically install compilers, upgrade dependencies, or download models.

## Node

**MOSS Model Loader**: Select backbone, Qwen3.5 text encoder, VAE, calculation accuracy and text device, output standard 'MODEL / CLIP / VAE'. The backbone and text encoder call the official 'load_model / load_clip', and VAE uses the ComfyUI original loader.

**MOSS Model Precompilation**: Receives 'MODEL / CLIP / VAE' output from an existing MOSS loader, suitable for continuing to use official standalone components or HF loading nodes.

Both nodes have the following options, all of which are turned off by default:

| Options | Role |
|---|---|
| Enable precompilation | Compile MOSS block peripheral element-by-element operations, preserving native Attention, MLP, and weight handling |
| Enable LoRA hotplugging | Subsequent normal LoRA nodes use compatible low-rank bypasses to support switching between requests |
| All precompiled (model computed) | Tried TE Transformer, DiT forward, and VAE encode/decode strict entire graphs separately, taking precedence over element-wise mode |

In normal use, the three loading nodes can be directly replaced, or the precompiled node can be connected to the original loader; the output can be connected to the LoRA, text encoding, sampling and VAE nodes as usual. No need to replace samplers or save nodes.

## Implementation

Element-by-element mode uses 'torch.compile / Inductor' to compile addition and multiplication separately, maintaining the modulation and residual order of the original block; normalization, projection, attention, nonlinearity and MLP continue to call the original implementation. The original model crops the real text token according to the mask without additional padding of the joint sequence. Method replacement is managed by 'ModelPatcher', and the original method is restored when uninstalling.

LoRA hotplug only accesses the standard LoRA loading API for tagged MOSS models. First parse the key and matrix dimensions using the original mapping, then execute 'base(x) + scale · up(down(x))', retaining the actual rank, not expanding AB, and not requantizing the base. Multiple adapters are injected in loading order, withdrawn in reverse order, and idle factors are moved back to the CPU. The text encoder follows the original strength rules; incompatible adapter types still use normal mounting.

Strict mode uses 'torch.compile(fullgraph=True, dynamic=True)' to build each model entry point according to the plan of the first actual input, reuse compatible versions, and do not enable CUDA Graph. It only wraps the entry point and does not check or replace the internal quantization operator; it does not support explicit error reporting when fully tracing, and does not silently convert to segmentation or eager.

## Boundary

- Native MOSS forward data dependency branching with mask and dynamic text cropping; current strict DiT full plots fail at this logic. Element-by-element mode preserves this part of the native flow and can be enabled independently.
- Dynamic VBAR residency does not support strict full-image. The original backend's quantization/FakeTensor support determines whether it is compileable; this node does not add patches for a certain format.
- Model entry map does not include word segmentation, sampling loops, scheduling, zooming in and file reading and writing. First-time compilation may be slow and does not guarantee speedup, arbitrary shape-free recompilation, or consistent image-by-value. Low-rank bypass and post-merge quantization may have numerical differences; concurrent parameter replacement during inference execution is not supported.

## Source code and verification

- ['backend.py'](backend.py): Reuse the official loading interface without registering official nodes repeatedly.
- ['runtime.py'](runtime.py): MOSS block element-by-element adaptation.
- ['full_precompile.py'](full_precompile.py): Strict compilation encapsulation without binding quantitative formatting.
- ['hot_lora.py'](hot_lora.py): Low-rank bypass and safe withdrawal.

CPU source/interface checking:

`bash
python scripts/check_source.py
python -m unittest discover -s tests -p "test_*.py"
`

Optional GPU checks:

`bash
python tests/verify_moss.py --comfy-dir "PATH_TO_COMFYUI"
python tests/verify_real_loader.py --comfy-dir "PATH_TO_COMFYUI" --dit-name "YOUR_DIT.safetensors" --text-encoder-name "YOUR_TE.safetensors" --vae-name "YOUR_VAE.safetensors"
`

'verify_moss.py' uses the official small random MOSS class to validate different sizes/mask, synthetic rank4/8 LoRA switching and recovery, and reports on strict mode compatibility; it does not load pretrained weights. 'verify_real_loader.py' verifies actual component loading and a small input native DiT forward, not a full out-of-the-picture test. Scripts support '--help', and results are written to '.test-output/'; CI only performs CPU checks.

## License

[GPL-3.0](LICENSE). The original model and loading API from [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)；MOSS block operational adaptation follows its Apache-2.0 implementation order. For upstream declarations, see [NOTICE](NOTICE) and [Apache-2.0](licenses/Apache-2.0.txt). The repository does not distribute weights or quantify the backend.
