
import utils.constants as constants
if constants.XLA_AVAILABLE:

    import torch_xla.distributed.spmd as xs


def print_sharding_info(tensor, name=None):
    
    if constants.XLA_AVAILABLE:
        xt = xs.wrap_if_sharded(tensor)

        print(f" ===== Sharding info for {name if name else 'tensor'} ===== ", flush=True)
        print(f" - Shape: {xt.shape}", flush=True)
        print(f" - Sharding spec: {xt.sharding_spec}\n", flush=True)

    else:
        raise RuntimeError("Cannot print sharding info. XLA is not available.")
    