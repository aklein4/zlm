import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch

from utils.checkpointing import (
    CheckpointManifest,
    TrainingCheckpointManager,
    canonical_parameter_name,
)


class CheckpointingTest(unittest.TestCase):

    def test_canonical_parameter_name_only_removes_wrapper_segments(self):
        self.assertEqual(
            canonical_parameter_name("model._orig_mod.layers.0.weight"),
            "model.layers.0.weight",
        )
        self.assertEqual(
            canonical_parameter_name("model.some_orig_mod.weight"),
            "model.some_orig_mod.weight",
        )

    def test_training_checkpoint_round_trip(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        model(torch.randn(4, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        expected = {
            name: value.clone()
            for name, value in model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            manager = TrainingCheckpointManager(checkpoint_dir)
            manager.save(
                1,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                {"global_step": 1, "lr_schedulers": {}},
                CheckpointManifest(global_step=1),
            )

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            manifest, trainer_state = manager.restore(1, state)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])

            self.assertEqual(manifest.global_step, 1)
            self.assertEqual(trainer_state["global_step"], 1)
            self.assertEqual(manager.latest_step(), 1)
            self.assertTrue(all(
                torch.equal(expected[name], model.state_dict()[name])
                for name in expected
            ))

    def test_checkpoint_rejects_mislabeled_step(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            manager = TrainingCheckpointManager(checkpoint_dir)
            with self.assertRaisesRegex(ValueError, "does not match"):
                manager.save(
                    2,
                    {},
                    {},
                    CheckpointManifest(global_step=1),
                )


if __name__ == "__main__":
    unittest.main()
