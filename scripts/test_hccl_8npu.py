import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.npu.set_device(local_rank)
dist.init_process_group(backend="hccl")

x = torch.tensor([rank + 1.0], device=f"npu:{local_rank}")
dist.all_reduce(x, op=dist.ReduceOp.SUM)

print(f"rank={rank}, local_rank={local_rank}, x={x.item()}")

dist.barrier()
dist.destroy_process_group()