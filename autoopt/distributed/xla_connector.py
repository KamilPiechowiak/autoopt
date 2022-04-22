from typing import Callable, List, Dict, Tuple

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from .base_connector import BaseConnector


class XlaConnector(BaseConnector):

    def get_device(self) -> torch.device:
        return xm.xla_device()

    def is_master(self) -> bool:
        return xm.is_master_ordinal()

    def rendezvous(self, name) -> None:
        xm.rendezvous(name)

    def get_samplers(self, train_dataset, val_dataset, shuffle_val=False) \
            -> Tuple[torch.utils.data.Sampler, torch.utils.data.Sampler]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=shuffle_val
        )
        return train_sampler, val_sampler

    def wrap_data_loader(self, data_loader, device) -> torch.utils.data.DataLoader:
        return pl.ParallelLoader(data_loader, [device]).per_device_loader(device)

    def optimizer_step(self, opt: torch.optim.Optimizer, **kwargs: Dict):
        xm.optimizer_step(opt, optimizer_args=self._filter_optimizer_kwargs(opt, kwargs))

    def reduce_gradients(self, opt: torch.optim.Optimizer) -> None:
        xm.reduce_gradients(opt)
        xm.mark_step()

    def all_avg(self, arr: List):
        cctx = xm.CollectiveContext()
        count = max(cctx.replica_devcount, cctx.world_size)
        xm.all_reduce(
            xm.REDUCE_SUM, arr, scale=1.0 / count, cctx=cctx)
        # xm.all_reduce(xm.REDUCE_SUM, arr, scale=1.0 / xm.xrt_world_size())

    def print(self, msg, flush=False):
        xm.master_print(msg, flush=flush)

    def save(self, obj: Dict, path: str):
        xm.save(obj, path)

    def run(self, function: Callable, args: List, nprocs: int):
        global SERIAL_EXEC
        SERIAL_EXEC = xmp.MpSerialExecutor()
        xmp.spawn(function, args=args, nprocs=nprocs, start_method='fork')

    def step(self):
        xm.mark_step()

    def all_gather(self, arr: List) -> List:
        arr = xm.all_gather(torch.tensor(arr))
        arr = [x.item() for x in arr]
        xm.mark_step()
        return arr
