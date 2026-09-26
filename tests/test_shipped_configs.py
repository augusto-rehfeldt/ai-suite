"""A fresh clone carries no *.local.json, so every provider must resolve to a shipped config."""
import unittest
from pathlib import Path

from ai_suite import providers, service


class ShippedConfigTests(unittest.TestCase):
    def test_every_provider_and_the_default_name_a_shipped_config(self):
        for name, path in {**providers.PROVIDER_CONFIG_MAP, "default": service.DEFAULT_CONFIG_PATH}.items():
            self.assertFalse(str(path).endswith(".local.json"), name)
            self.assertTrue(Path(path).is_file(), name)

    def test_shipped_configs_carry_no_key(self):
        import json
        for path in Path(providers.PACKAGE_ROOT, "config").glob("ai_config_*.json"):
            if not path.name.endswith(".local.json"):
                self.assertFalse(json.loads(path.read_text(encoding="utf-8")).get("api_key"), path.name)


if __name__ == "__main__":
    unittest.main()
