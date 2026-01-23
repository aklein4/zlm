import torch

import utils.constants as constants
if constants.XLA_AVAILABLE:

    import torch_xla.distributed.spmd as xs


def shard_with_gradients(
    x: torch.Tensor,
    mesh = None,
    spec = None,
) -> torch.Tensor:
    if not constants.XLA_AVAILABLE:
        raise RuntimeError(
            "XLA is not available, cannot shard tensor."
        )

    if mesh is None:
        mesh = xs.get_global_mesh()
        assert mesh is not None, "No sharding mesh found."

    if spec is None:
        spec = constants.BATCH_SHARD_SPEC

    return xs.mark_sharding_with_gradients(
        x, mesh, spec
    )


def maybe_shard_with_gradients(
    x: torch.Tensor,
    mesh = None,
    spec = None,
) -> torch.Tensor:
    if constants.XLA_AVAILABLE:
        return shard_with_gradients(x, mesh, spec)
    else:
        return x


def replicated_optimizer_step(
    optimizer: torch.optim.Optimizer,
):
    if not constants.XLA_AVAILABLE:
        raise RuntimeError(
            "XLA is not available, cannot perform replicated optimizer step."
        )

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:

                

                p.grad = xs.all_reduce(
                    p.grad,
                    mesh=xs.get_global_mesh(),
                    reduce_op="sum",
                )   

    return optimizer.step()