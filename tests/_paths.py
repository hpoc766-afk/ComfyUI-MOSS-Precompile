# SPDX-License-Identifier: GPL-3.0-only
"""Shared CLI for optional GPU checks; no imports before argument parsing."""
import argparse
from pathlib import Path
from types import SimpleNamespace


def configure(name):
    parser=argparse.ArgumentParser(description='MOSS/ComfyUI validation in an existing compatible CUDA environment.')
    parser.add_argument('--comfy-dir',required=True,type=Path)
    parser.add_argument('--moss-dir',type=Path,help='Official moss-image package; defaults to COMFY/custom_nodes/moss-image')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--dit-name',default='moss-image-base-dit.safetensors')
    parser.add_argument('--text-encoder-name',default='moss-image-base-text-encoder.safetensors')
    parser.add_argument('--vae-name',default='moss-image-base-vae.safetensors')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];comfy=args.comfy_dir.resolve()
    if not (comfy/'main.py').is_file():parser.error('--comfy-dir must contain main.py')
    moss=args.moss_dir.resolve() if args.moss_dir else comfy/'custom_nodes/moss-image'
    if not (moss/'model.py').is_file():parser.error('--moss-dir must contain the official model.py')
    out=args.output_dir.resolve() if args.output_dir else root/'.test-output'/name
    out.mkdir(parents=True,exist_ok=True)
    return SimpleNamespace(root=root,comfy=comfy,moss=moss,out=out,args=args)
