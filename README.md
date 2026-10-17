# ComfyUI-MOSS-Precompile

为 MOSS-Image 提供模型加载、可选逐元素预编译和 LoRA 热切换。权重解析与计算后端沿用官方 MOSS 节点，编译层不依赖特定量化格式，不展开权重，也不注册格式专用算子。

## 安装

先按 [MOSS-Image 官方说明](https://github.com/OpenMOSS/MOSS-Image/tree/main/comfyui) 安装 `moss-image` 后端，再在 ComfyUI 目录执行：

```bash
git clone https://github.com/hpoc766-afk/ComfyUI-MOSS-Precompile.git custom_nodes/ComfyUI-MOSS-Precompile
```

重启 ComfyUI 后使用。更新：

```bash
git -C custom_nodes/ComfyUI-MOSS-Precompile pull --ff-only
```

使用已有且相互兼容的 ComfyUI、PyTorch 和模型后端。开启预编译还需要可用的 Inductor 编译环境；本节点不自动安装编译器、升级依赖或下载模型。

## 节点

**MOSS 模型加载器**：选择主干、Qwen3.5 文本编码器、VAE、计算精度与文本设备，输出标准 `MODEL / CLIP / VAE`。主干与文本编码器调用官方 `load_model / load_clip`，VAE 使用 ComfyUI 原加载器。

**MOSS 模型预编译**：接收已有 MOSS 加载器输出的 `MODEL / CLIP / VAE`，适合继续使用官方独立组件或 HF 加载节点。

两个节点都有以下选项，默认全部关闭：

| 选项 | 作用 |
|---|---|
| 启用预编译 | 编译 MOSS block 外围逐元素运算，保留原生 Attention、MLP 和权重处理 |
| 启用 LoRA 热插拔 | 后续普通 LoRA 节点使用兼容低秩旁路，支持请求之间切换 |
| 全部预编译（模型计算） | 分别尝试 TE Transformer、DiT 前向和 VAE encode/decode 严格整图，优先于逐元素模式 |

普通使用可直接替换三个加载节点，或将预编译节点接在原加载器后；输出照常连接 LoRA、文本编码、采样及 VAE 节点。无需更换采样器或保存节点。

## 实现方式

逐元素模式使用 `torch.compile / Inductor` 分别编译加法和乘法，保持原 block 的调制与残差顺序；归一化、投影、Attention、非线性和 MLP 继续调用原实现。原模型按 mask 裁剪真实文本 token，不额外填充联合序列。方法替换由 `ModelPatcher` 管理，卸载时恢复原方法。

LoRA 热插拔仅对标记的 MOSS 模型接入标准 LoRA 加载 API。先用原映射解析 key 和矩阵维度，再执行 `base(x) + scale · up(down(x))`，保留实际 rank、不展开 AB、不重新量化基座。多个适配器按加载顺序注入、反序撤回，闲置因子移回 CPU。文本编码器沿用原强度规则；不兼容的适配器类型仍走普通挂载。

严格模式使用 `torch.compile(fullgraph=True, dynamic=True)`，按首次实际输入建立各模型入口的计划，复用兼容版本，不启用 CUDA Graph。它只包装入口，不检查或替换内部量化算子；不支持完整追踪时明确报错，不静默转为分段或 eager。

## 边界

- 原生 MOSS 前向含 mask 的数据依赖分支和动态文本裁剪；当前严格 DiT 整图会在该逻辑处失败。逐元素模式保留这部分原生流程，可独立启用。
- 动态 VBAR 驻留不支持严格整图。原后端的量化／FakeTensor 支持决定是否可编译；本节点不为某种格式增加补丁。
- 模型入口图不包含分词、采样循环、调度、放大与文件读写。首编译可能较慢，不保证加速、任意形状免重编译或画面逐值一致。低秩旁路与合并后量化可能有数值差异；不支持推理执行中途并发替换参数。

## 源码与验证

- [`backend.py`](backend.py)：复用官方加载接口，不重复注册官方节点。
- [`runtime.py`](runtime.py)：MOSS block 逐元素适配。
- [`full_precompile.py`](full_precompile.py)：不绑定量化格式的严格编译封装。
- [`hot_lora.py`](hot_lora.py)：低秩旁路与安全撤回。

CPU 源码／接口检查：

```bash
python scripts/check_source.py
python -m unittest discover -s tests -p "test_*.py"
```

可选 GPU 检查：

```bash
python tests/verify_moss.py --comfy-dir "PATH_TO_COMFYUI"
python tests/verify_real_loader.py --comfy-dir "PATH_TO_COMFYUI" --dit-name "YOUR_DIT.safetensors" --text-encoder-name "YOUR_TE.safetensors" --vae-name "YOUR_VAE.safetensors"
```

`verify_moss.py` 使用官方的小型随机 MOSS 类，验证不同尺寸／mask、合成 rank4/8 LoRA 切换与恢复，并报告严格模式是否兼容；它不加载预训练权重。`verify_real_loader.py` 验证实际组件加载和一次小输入原生 DiT 前向，不是完整出图测试。脚本支持 `--help`，结果写入 `.test-output/`；CI 仅执行 CPU 检查。

## 许可证

[GPL-3.0](LICENSE)。原模型和加载 API 来自 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)；MOSS block 运算适配遵循其 Apache-2.0 实现顺序。上游声明见 [NOTICE](NOTICE) 与 [Apache-2.0](licenses/Apache-2.0.txt)。仓库不分发权重或量化后端。
