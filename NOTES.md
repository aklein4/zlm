
Notes about how this repo and jax in general works.

# Workflow

1. Enter main() through apsl app
 - For argument passing?

2. initialize()

3. run()
 - train_loop wrapped in diagnostic(?) contexts

4. setup_train_loop()
 - create model using model_creation_utils.from_config()
 - create_training_utils()
 - create_data_iterator()
 - create_rampup_manager()
 - create_dataloader()
 - apply mesh to data_iterator
 - maxtext_utils.setup_training_state()
 - sharding.assert_params_sufficiently_sharded()
 - debug sharding


# Details

## layers

### models

 - Transformer
     - creates embedding
     - creates decoder using nnx_wrappers.ToNNX
     - decoder.lazy_init

### Llama2
 - defines a LlamaDecoderLayer nnx.Module

### nnx_wrappers
 - TODO

### decoders

 - SequentialBlockDecoderLayers

 - Decoder

## model_creation_utils

 - from_config()
     - create mesh
     - get model from create_model()

 - get_transformer_model()
     - calls models.Transformer(config...)

 - create_model()
     - wraps get_transformer_model() in quantization

 - create_nnx_model()
     - unused?
