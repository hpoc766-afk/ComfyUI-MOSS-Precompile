# SPDX-License-Identifier: GPL-3.0-only
"""Loader/adapter options preserve the official backend contract, without torch."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]


def node_module():
    name='_moss_node_contract'
    spec=importlib.util.spec_from_file_location(name,ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
    module=importlib.util.module_from_spec(spec)
    model,clip,vae=object(),object(),object()
    backend=types.SimpleNamespace(DTYPES={'default':None,'bf16':'bf16'},load_model=Mock(return_value=model),load_clip=Mock(return_value=clip))
    partial=Mock(return_value='partial');full=Mock(return_value=('full-m','full-c','full-v'));hot=Mock(side_effect=lambda x:('hot',x))
    is_moss=Mock(return_value=True)
    folders=types.SimpleNamespace(get_filename_list=lambda key:['example.safetensors'],get_full_path_or_raise=lambda kind,file:kind+'/'+file)
    vae_loader=type('VAE',(),{'vae_list':lambda self:['vae.safetensors'],'load_vae':lambda self,name:(vae,)})
    stubs={name:module,'folder_paths':folders,'nodes':types.SimpleNamespace(VAELoader=vae_loader),
       name+'.backend':types.SimpleNamespace(get_backend=lambda:backend),
       name+'.runtime':types.SimpleNamespace(patch_moss=partial),
       name+'.full_precompile':types.SimpleNamespace(patch_full_models=full),
       name+'.hot_lora':types.SimpleNamespace(enable_hot_lora=hot),
       name+'.model_kind':types.SimpleNamespace(is_moss=is_moss)}
    with patch.dict(sys.modules,stubs):spec.loader.exec_module(module)
    return module,backend,model,clip,vae,partial,full,hot,is_moss


class LoaderContract(unittest.TestCase):
    def test_native_loading_is_default(self):
        n,b,m,c,v,partial,full,hot,_=node_module()
        self.assertEqual(n.MOSSPrecompileLoader().load('dit','te','vae'),(m,c,v))
        b.load_model.assert_called_once_with('diffusion_models/dit','default')
        b.load_clip.assert_called_once_with('text_encoders/te','default')
        partial.assert_not_called();full.assert_not_called();hot.assert_not_called()
        for kind in n.NODE_CLASS_MAPPINGS.values():
            self.assertEqual(kind.RETURN_TYPES,('MODEL','CLIP','VAE'))
            self.assertTrue(all(not item[1]['default'] for item in kind.INPUT_TYPES()['optional'].values()))

    def test_dtype_and_text_device_go_to_original_loader(self):
        n,b,*_=node_module()
        n.MOSSPrecompileLoader().load('arbitrary-format','te','vae','bf16','cpu')
        b.load_model.assert_called_once_with('diffusion_models/arbitrary-format','bf16')
        b.load_clip.assert_called_once_with('text_encoders/te','cpu')

    def test_unsupported_precision_rejected_before_loading(self):
        n,b,*_=node_module()
        with self.assertRaises(ValueError):n.MOSSPrecompileLoader().load('dit','te','vae','unknown')
        b.load_model.assert_not_called();b.load_clip.assert_not_called()

    def test_existing_models_pass_through_when_off(self):
        n,b,m,c,v,partial,full,hot,_=node_module()
        self.assertEqual(n.MOSSPrecompile().apply(m,c,v),(m,c,v))
        b.load_model.assert_not_called();partial.assert_not_called();full.assert_not_called();hot.assert_not_called()

    def test_partial_compilation_and_hot_lora_are_independent(self):
        n,b,m,c,v,partial,full,hot,_=node_module()
        self.assertEqual(n.MOSSPrecompile().apply(m,c,v,precompile=True,hot_lora=True),(('hot','partial'),c,v))
        partial.assert_called_once_with(m);hot.assert_called_once_with('partial');full.assert_not_called()

    def test_full_mode_wins_without_changing_hot_option(self):
        n,b,m,c,v,partial,full,hot,_=node_module()
        self.assertEqual(n.MOSSPrecompile().apply(m,c,v,True,True,True),(('hot','full-m'),'full-c','full-v'))
        full.assert_called_once_with(m,c,v);partial.assert_not_called();hot.assert_called_once_with('full-m')

    def test_non_moss_model_rejected_before_patching(self):
        n,b,m,c,v,partial,full,hot,kind=node_module();kind.return_value=False
        with self.assertRaises(ValueError):n.MOSSPrecompile().apply(m,c,v,True,True,True)
        partial.assert_not_called();full.assert_not_called();hot.assert_not_called()


if __name__=='__main__':unittest.main()
