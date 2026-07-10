import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hydra import compose, initialize_config_dir


class ConfigurationTest(unittest.TestCase):

    def test_default_config_composes_with_explicit_checkpoint_strictness(self):
        config_dir = str((Path(__file__).parents[1] / "src" / "configs").resolve())
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="default")

        self.assertIs(config.checkpoint.initialize_strict, True)
        self.assertIs(config.model.pretrained_strict, True)
        self.assertGreater(config.trainer.checkpoint_interval, 0)
        self.assertEqual(config.model.type, "llama.LlamaForCausalLM")


if __name__ == "__main__":
    unittest.main()
