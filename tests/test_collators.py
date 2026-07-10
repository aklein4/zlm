import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from collators.seq_to_seq import SeqToSeqCollator
from collators.single_sequence import SingleSequenceCollator


class CollatorTest(unittest.TestCase):

    def test_single_sequence_collator_preserves_external_pad_id(self):
        collator = SingleSequenceCollator(
            sequence_length=4,
            pad_token_id=32,
            vocab_size=32,
        )
        batch = collator([{"input_ids": [1, 2]}])

        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 32, 32]])
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[True, True, False, False]],
        )
        self.assertEqual(batch["labels"].tolist(), [[1, 2, -100, -100]])

    def test_seq_to_seq_collator_returns_standard_masks_and_labels(self):
        collator = SeqToSeqCollator(
            input_length=3,
            output_length=4,
            pad_token_id=32,
        )
        batch = collator([{"input_ids": [1, 2], "output_ids": [3]}])

        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, False]])
        self.assertEqual(
            batch["decoder_attention_mask"].tolist(),
            [[True, False, False, False]],
        )
        self.assertEqual(batch["labels"].tolist(), [[3, -100, -100, -100]])


if __name__ == "__main__":
    unittest.main()
