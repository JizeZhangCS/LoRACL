import torch
import torch.distributed as dist

@torch.no_grad()
def concat_all_gather_interlace(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    inds = [torch.arange(i, len(g)*len(tensors_gather), len(tensors_gather)) for i, g in enumerate(tensors_gather)]
    inds = torch.cat(inds, dim=0)
    sort_inds = torch.argsort(inds)
    return output[sort_inds]