# Transformers compatibility

The public model classes use `PretrainedConfig`, `PreTrainedModel`, standard
`ModelOutput` objects, loss-first tuple outputs, `attention_mask`, and labels
with `-100` ignored positions. The internal model cores retain the fixed tensor
structures used by XLA scan and compiled training.

Register the local implementations before using AutoClasses:

```python
from models import register_transformers_auto_classes
from transformers import AutoModelForCausalLM

register_transformers_auto_classes()
model = AutoModelForCausalLM.from_pretrained("path/to/export")
```

`LlamaForCausalLM` supports standard no-cache `generate()`. The custom Llama
decoder additionally accepts Transformers `Cache` objects. TPU generation uses
a fixed-size `StaticCache`, fixed input shapes, and an XLA step boundary between
decode iterations to avoid an ever-growing lazy graph.

The configured padding ID may intentionally sit immediately outside the output
vocabulary for compatibility with existing checkpoints. Public model calls must
therefore include an `attention_mask`; masked IDs are replaced before embedding.
Collators supplied by this repository always emit that mask and convert padded
labels to `-100`.

Frequent TPU recovery checkpoints should remain in DCP format. Call portable
export only for checkpoints that need Transformers, evaluation, or Hub access;
standard `save_pretrained()` must not independently gather the model on every
TPU rank.
