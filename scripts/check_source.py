# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only source, localization and CLI checks; no ComfyUI/torch dependency."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    sources=[p for p in ROOT.rglob('*.py') if not any(v in {'.git','.test-output','__pycache__','.venv','venv'} for v in p.relative_to(ROOT).parts)]
    for path in sources:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    expected={'MOSSPrecompileLoader':{'dit_name','text_encoder_name','vae_name','weight_dtype','text_device','precompile','hot_lora','full_precompile'},
              'MOSSPrecompile':{'model','clip','vae','precompile','hot_lora','full_precompile'}}
    translations=[]
    for language in ['zh','zh-CN']:
        data=json.loads((ROOT/'locales'/language/'nodeDefs.json').read_text(encoding='utf-8'))
        assert set(data)==set(expected)
        for key,fields in expected.items():
            assert set(data[key]['inputs'])==fields
            assert set(data[key]['outputs'])=={'0','1','2'}
        translations.append(data)
    assert translations[0]==translations[1]
    for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(ROOT/'README.md').read_text(encoding='utf-8')):
        target=raw.strip('<>').split('#',1)[0]
        if target and '://' not in target:assert (ROOT/target).exists(),target
    tests=list((ROOT/'tests').glob('verify_*.py'))
    for path in tests:
        process=subprocess.run([sys.executable,str(path),'--help'],capture_output=True,text=True,timeout=15)
        assert process.returncode==0,(path.name,process.stderr)
        assert '--comfy-dir' in process.stdout and '--output-dir' in process.stdout
    assert 'GNU GENERAL PUBLIC LICENSE' in (ROOT/'LICENSE').read_text(encoding='utf-8')
    assert 'Apache License' in (ROOT/'licenses/Apache-2.0.txt').read_text(encoding='utf-8')
    # The generic compiler must not include layout-specific access or kernels.
    compiler=(ROOT/'full_precompile.py').read_text(encoding='utf-8')
    for forbidden in ['comfy_kitchen','_qdata','_params','custom_op','allow_in_graph','ConvRot','convrot']:
        assert forbidden not in compiler,forbidden
    print(f'PASS: {len(sources)} Python sources, two node locales, {len(tests)} CLI help interfaces, links and licenses checked; generic compiler has no quantization bridge.')
    print('CPU checks only; no GPU inference.')

if __name__=='__main__':main()
