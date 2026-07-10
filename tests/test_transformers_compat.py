import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch
from transformers import AutoModel, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from models import register_transformers_auto_classes
from models.configuration import TPULlamaConfig, ZLMConfig
from models.custom_llama import CustomLlamaForCausalLM
from models.llama import LlamaForCausalLM
from models.modeling_outputs import ZLMCausalLMOutput
from models.zlm import ZLMModel


def llama_config():
    return TPULlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_kernel=None,
        pad_token_id=32,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=32,
    )


def zlm_config():
    return ZLMConfig(
        **llama_config().to_dict(),
        input_length=3,
        output_length=4,
        z_length=4,
        latent_size=4,
        z_ar_steps=2,
        head_intermediate_size=16,
        pretrained_llama=None,
    )


class TransformersCompatibilityTest(unittest.TestCase):

    def test_causal_lm_uses_transformers_outputs_and_padding_contract(self):
        model = LlamaForCausalLM(llama_config())
        input_ids = torch.tensor([[1, 2, 3, 32]])
        attention_mask = input_ids != 32
        labels = torch.where(attention_mask, input_ids, -100)

        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        self.assertIsInstance(output, CausalLMOutputWithPast)
        self.assertEqual(output.logits.shape, (1, 4, 32))
        self.assertTrue(output.loss.isfinite())

        tuple_output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )
        self.assertEqual(tuple_output[0].ndim, 0)
        self.assertEqual(tuple_output[1].shape, output.logits.shape)

    def test_causal_lm_auto_class_round_trip_and_generation(self):
        register_transformers_auto_classes()
        model = LlamaForCausalLM(llama_config())
        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir)
            restored = AutoModelForCausalLM.from_pretrained(save_dir)

        self.assertIsInstance(restored, LlamaForCausalLM)
        self.assertTrue(torch.equal(restored.lm_head.weight, model.lm_head.weight))

        output_ids = restored.generate(
            torch.tensor([[1, 2]]),
            max_new_tokens=2,
            min_new_tokens=2,
            do_sample=False,
        )
        self.assertEqual(output_ids.shape, (1, 4))

    def test_zlm_auto_class_round_trip(self):
        register_transformers_auto_classes()
        model = ZLMModel(zlm_config())
        input_ids = torch.tensor([[1, 2, 32]])
        attention_mask = input_ids != 32
        labels = torch.tensor([[3, 4, 5, -100]])

        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            noise_scale=0.0,
        )
        self.assertIsInstance(output, ZLMCausalLMOutput)
        self.assertEqual(output.logits.shape, (1, 4, 32))
        self.assertTrue(output.loss.isfinite())

        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir)
            restored = AutoModel.from_pretrained(save_dir)

        self.assertIsInstance(restored, ZLMModel)
        self.assertTrue(torch.equal(restored.lm_head.weight, model.lm_head.weight))

    def test_custom_llama_static_cache_has_fixed_shape_decode(self):
        config = llama_config()
        config.pad_token_id = 0
        config.use_cache = True
        model = CustomLlamaForCausalLM(config)
        cache = StaticCache(config, max_cache_len=8)

        model(
            torch.tensor([[1, 2]]),
            past_key_values=cache,
            cache_position=torch.tensor([0, 1]),
            use_cache=True,
        )
        output = model(
            torch.tensor([[3]]),
            position_ids=torch.tensor([[2]]),
            past_key_values=cache,
            cache_position=torch.tensor([2]),
            use_cache=True,
        )

        self.assertEqual(cache.get_seq_length(), 3)
        self.assertEqual(output.logits.shape, (1, 1, 32))
        self.assertTrue(output.logits.isfinite().all())

        generated = model.generate(
            torch.tensor([[1, 2]]),
            max_new_tokens=2,
            min_new_tokens=2,
            cache_implementation="static",
            do_sample=False,
        )
        self.assertEqual(generated.shape, (1, 4))

        register_transformers_auto_classes()
        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir)
            restored = AutoModelForCausalLM.from_pretrained(save_dir)
        self.assertIsInstance(restored, CustomLlamaForCausalLM)


if __name__ == "__main__":
    unittest.main()
