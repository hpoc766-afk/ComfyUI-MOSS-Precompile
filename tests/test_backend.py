# SPDX-License-Identifier: GPL-3.0-only
"""Backend lookup must reuse official modules and must not register nodes twice."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]


def load_module():
    spec=importlib.util.spec_from_file_location('_backend_contract',ROOT/'backend.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class BackendContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.moss=self.root/'moss-image';self.moss.mkdir()
        (self.moss/'model.py').write_text('VALUE = 17\n',encoding='utf-8')
        (self.moss/'__init__.py').write_text('raise RuntimeError("Extension registration must not run twice")\n',encoding='utf-8')
        (self.moss/'loaders.py').write_text('from .model import VALUE\nDTYPES={"default":None}\ndef load_model(*args): return VALUE\ndef load_clip(*args): return VALUE\n',encoding='utf-8')

    def test_loaded_official_loader_is_reused(self):
        backend=load_module()
        loaded=types.ModuleType('_existing_moss_loaders');loaded.__file__=str(self.moss/'loaders.py')
        loaded.load_model=lambda *x:17;loaded.load_clip=lambda *x:17
        with patch.dict(sys.modules,{'_existing_moss_loaders':loaded,'folder_paths':types.SimpleNamespace(get_folder_paths=lambda _: [str(self.root)])}):
            self.assertIs(backend.get_backend(),loaded)

    def test_discovery_order_does_not_import_extension_entrypoint(self):
        backend=load_module();before=set(sys.modules)
        try:
            with patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(get_folder_paths=lambda _: [str(self.root)])}):
                loaded=backend.get_backend();self.assertEqual(loaded.load_model(),17)
                self.assertIs(backend.get_backend(),loaded)
        finally:
            for name in set(sys.modules)-before:
                if name.startswith('_comfy_moss_precompile_backend_'):sys.modules.pop(name,None)

    def test_missing_backend_has_actionable_error(self):
        backend=load_module()
        with patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(get_folder_paths=lambda _: [])}):
            with self.assertRaisesRegex(RuntimeError,'moss-image'):backend.get_backend()

if __name__=='__main__':unittest.main()
