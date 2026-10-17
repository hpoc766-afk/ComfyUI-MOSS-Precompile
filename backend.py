# SPDX-License-Identifier: GPL-3.0-only
"""Resolve the installed official MOSS backend; do not vendor model/quant code."""
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types


def backend_directories():
    import folder_paths
    roots=list(folder_paths.get_folder_paths("custom_nodes"))
    for root in roots:
        candidate=Path(root)/"moss-image"
        if (candidate/"loaders.py").is_file() and (candidate/"model.py").is_file():
            yield candidate.resolve()


def get_backend():
    for root in backend_directories():
        expected=root/"loaders.py"
        # Reuse the official module when ComfyUI has already loaded it, even if
        # its dynamic package name differs across ComfyUI versions.
        for _,module in list(sys.modules.items()):
            file=getattr(module,"__file__",None)
            if isinstance(file,str) and Path(file).resolve()==expected:
                if callable(getattr(module,"load_model",None)) and callable(getattr(module,"load_clip",None)):
                    return module
        # No dependency on discovery order. Import only the official loader and
        # its relative dependencies, not a second extension/node registration.
        name="_comfy_moss_precompile_backend_"+str(root).encode().hex()
        if name not in sys.modules:
            package=types.ModuleType(name)
            package.__file__=str(root/"__init__.py")
            package.__package__=name;package.__path__=[str(root)]
            package.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
            sys.modules[name]=package
        return importlib.import_module(name+".loaders")
    raise RuntimeError(
        "未找到官方moss-image后端。请先安装OpenMOSS/MOSS-Image提供的ComfyUI节点，"
        "使custom_nodes/moss-image包含loaders.py与model.py。"
    )
