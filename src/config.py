import os, yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Cfg:
    d: Dict[str, Any]
    def __getattr__(self, k):
        v = self.d.get(k)
        if isinstance(v, dict):
            return Cfg(v)
        return v

    def get(self, k, default=None):
        return self.d.get(k, default)

    @staticmethod
    def load(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        return Cfg(d)

    def dump(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.d, f, sort_keys=False, allow_unicode=True)