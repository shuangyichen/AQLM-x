from __future__ import annotations

import math
import random
from argparse import Namespace
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import Gather

from src.aq import QuantizedWeight
from src.utils import ellipsis

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.decomposition import PCA 
_layer_counter = 0 


def next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def fast_walsh_hadamard_transform(X):
    """Vectorized FWHT — no Python loops over elements."""
    d = X.shape[-1]
    assert (d & (d - 1)) == 0, "Dimension must be a power of 2"
    X = X.clone()
    h = 1
    while h < d:
        X = X.view(*X.shape[:-1], d // (2 * h), 2, h)
        x0 = X[..., 0, :].clone()
        x1 = X[..., 1, :].clone()
        X[..., 0, :] = x0 + x1
        X[..., 1, :] = x0 - x1
        X = X.view(*X.shape[:-3], d)
        h *= 2
    return X / (d ** 0.5)


class AQEngine(nn.Module):
    """A wrapper class that runs AQ training for a single linear layer. All the important math is in aq.py"""

    def __init__(self, layer: nn.Linear, accumulator_dtype: torch.dtype = torch.float64):
        super().__init__()
        self.layer = layer
        self.device = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.register_buffer(
            "XTX", torch.zeros((self.columns, self.columns), dtype=accumulator_dtype, device=self.device)
        )
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """Accumulate a minibatch of layer inputs and update the X.T @ X (aka half hessian)"""
        assert self.XTX is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        tmp = inp.shape[0]
        inp = inp.t()

        self.XTX *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(1 / self.nsamples) * inp.to(self.XTX.dtype)
        self.XTX += inp.matmul(inp.t())


    @torch.enable_grad()
    def quantize(self, *, args: Namespace, verbose: bool = True) -> QuantizedWeight:
        """create a QuantizedLinear with specified args based on the collected hessian (XTX) data"""
        assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
        assert args.devices[0] == self.device, (args.devices[0], self.XTX.device)

        ### For the Block FWHT
        d_in = self.columns
        self.W_raw = self.layer.weight.data.clone().float()

        # Find largest power of 2 that divides d_in exactly, capped at 256
        BLOCK = 1
        for b in [256, 128, 64, 32]:
            if d_in % b == 0:
                BLOCK = b
                break
        self.block_size = BLOCK
        self.signs = torch.randint(0, 2, (d_in,), device=self.device) * 2 - 1

        if BLOCK > 1:
            W_signed = self.W_raw * self.signs.unsqueeze(0)          # [out, d_in]
            W_blocks = W_signed.view(-1, d_in // BLOCK, BLOCK)       # [out, nblocks, BLOCK]
            W_rotated = fast_walsh_hadamard_transform(W_blocks).view(-1, d_in)
            self.layer.weight.data = W_rotated.to(dtype=self.layer.weight.dtype)

            XTX_raw = self.XTX.clone().float()
            XTX_signed = XTX_raw * self.signs.unsqueeze(0) * self.signs.unsqueeze(1)
            XTX_out = torch.zeros_like(XTX_signed)
            for i in range(d_in // BLOCK):
                s, e = i * BLOCK, (i + 1) * BLOCK
                block = XTX_signed[s:e, s:e]
                XTX_out[s:e, s:e] = fast_walsh_hadamard_transform(
                    fast_walsh_hadamard_transform(block).T
                ).T
            self.XTX = XTX_out.to(dtype=self.XTX.dtype)
            # if verbose:
            #     print(f"[Rotation Engine] Applied block-Hadamard (d_in={d_in}, BLOCK={BLOCK}, nblocks={d_in // BLOCK}).")
        else:
            self.signs = None
            # if verbose:
            #     print(f"[Rotation Engine] Skipped — d_in={d_in} not divisible by any supported block size.")
        ###

        self.quantized_weight = QuantizedWeight(
            reference_weight=self.layer.weight.detach().to(device=self.device, dtype=torch.float32),
            out_group_size=args.out_group_size,
            in_group_size=args.in_group_size,
            num_codebooks=args.num_codebooks,
            nbits_per_codebook=args.nbits_per_codebook,
            codebook_value_nbits=args.codebook_value_nbits,
            codebook_value_num_groups=args.codebook_value_num_groups,
            scale_nbits=args.scale_nbits,
            max_iter=args.init_max_iter,
            max_points_per_centroid=args.init_max_points_per_centroid,
            devices=args.devices,
            verbose=True,
        )   
        
        differentiable_parameters = nn.ParameterDict(
            {name: param for name, param in self.quantized_weight.named_parameters() if param.requires_grad}
        )
        opt = torch.optim.Adam(differentiable_parameters.values(), lr=args.lr, betas=(0.0, 0.95), amsgrad=True)

        replicas = None
        if len(args.devices) > 1:
            replicas = torch.nn.parallel.replicate(self, args.devices)
            replicas[0] = self

        previous_best_loss = float("inf")  # for early stopping
        for epoch in range(args.max_epochs):
            # train codebooks and scales
            for step in range(args.steps_per_epoch):
                if len(args.devices) == 1:
                    loss = self._compute_mse()
                else:
                    loss = self._compute_mse_parallel(args.devices, replicas, differentiable_parameters)

                if not torch.isfinite(loss).item():
                    raise ValueError(f"Quantization loss is {loss}")
                if step == 0 and args.relative_mse_tolerance is not None:
                    if loss.item() / previous_best_loss > (1.0 - args.relative_mse_tolerance):
                        return self.quantized_weight  # early stopping; no updates after last epoch's beam search
                    previous_best_loss = min(previous_best_loss, loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and (epoch * args.steps_per_epoch + step) % args.print_frequency == 0:
                    print(f"epoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")

            # search for better codes (cluster indices)
            seed = random.getrandbits(256)
            self.beam_search_update_codes_(
                args.devices,
                replicas,
                differentiable_parameters,
                seed=seed,
                beam_size=args.beam_size,
                verbose=True,
            )

        return self.quantized_weight

    # modified for DropbyDrop
    def _compute_mse(self, selection: Union[slice, ellipsis] = ...) -> torch.Tensor:
        """
        Compute the activation MSE error = ||X @ quantized_weight - X @ reference_weight||^2
        Use the square-of-difference formula to avoid materializing per-batch predictions
        :param selection:  By default, compute MSE normally. If selection is specified, this method will instead
            compute MSE over a portion of output channels that align with the selected out_groups (for parallelism)
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        """
        assert self.quantized_weight is not None, "必须在 AQUtil.quantize 内部/之后调用"
    
        if isinstance(selection, ellipsis):
            reference_weight = self.layer.weight.detach().to(self.quantized_weight.codebooks.dtype)
        else:
            assert isinstance(selection, slice)
            out_channel_selection = slice(
                selection.start * self.quantized_weight.out_group_size,
                selection.stop * self.quantized_weight.out_group_size,
            )
            reference_weight = self.layer.weight.detach()[out_channel_selection].to(self.quantized_weight.codebooks.dtype)
        
        total_codebooks = self.quantized_weight.num_codebooks
               
        #gemma-weights
        #codebook_weights = torch.tensor([1000,  1000, 10,   0.1, 0.1], device=self.device, dtype=self.XTX.dtype)

        #gemma-weights2 
        #codebook_weights = torch.tensor([1000,  1000, 1000, 0.1, 0.1], device=self.device, dtype=self.XTX.dtype)

        #Uni
        #codebook_weights = torch.tensor([0.2,0.2, 0.2, 0.2, 0.2], device=self.device, dtype=self.XTX.dtype)

        #35W 
        codebook_weights = torch.tensor([0,0, 0.5,  0, 0.5], device=self.device, dtype=self.XTX.dtype)

        #34W 
        #codebook_weights = torch.tensor([0,0, 0.5, 0.5], device=self.device, dtype=self.XTX.dtype)

        #234W 
        #codebook_weights = torch.tensor([0,0.333, 0.333, 0.333], device=self.device, dtype=self.XTX.dtype)

        #24W 
        #codebook_weights = torch.tensor([0,0.5, 0, 0.5], device=self.device, dtype=self.XTX.dtype)

        #345W
        #codebook_weights = torch.tensor([0,0, 0.333,  0.333, 0.333], device=self.device, dtype=self.XTX.dtype)

        total_loss = torch.tensor(0.0, device=self.device, dtype=self.XTX.dtype)
        
        for i in range(1, total_codebooks + 1):
            
            # Skip if weight is zero 
            codebook_weight = codebook_weights[i - 1]
            if codebook_weight.item() == 0: 
                continue
            quantized_weight_i = self.quantized_weight(selection, num_codebooks=i)
            
            delta_weight = (quantized_weight_i - reference_weight).to(self.XTX.dtype)
            mse_i = (delta_weight @ self.XTX).flatten() @ delta_weight.flatten() / self.quantized_weight.out_features

            # Ensure all tensors are on the same device before computation
            mse_i = mse_i.to(self.device)
            codebook_weight = codebook_weight.to(self.device)
            
            #total_loss = total_loss + mse_i
            total_loss = total_loss + codebook_weight* mse_i 

        return total_loss

    def _replace_and_compute_mse(self, params_to_replace: nn.ParameterDict, selection: slice) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then compute MSE"""
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        return self._compute_mse(selection)

    def _compute_mse_parallel(
        self, devices: Sequence[torch.device], replicas: Sequence[AQEngine], parameters_to_replicate: nn.ParameterDict
    ) -> torch.Tensor:
        """Compute MSE in parallel over output channels"""
        replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
        num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
        shard_size = (num_output_groups - 1) // len(devices) + 1
        active_slices_by_replica = [
            slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
        ]
        funcs_by_replica = [replica._replace_and_compute_mse for replica in replicas]
        inputs_by_replica = [(dict(), active_slices_by_replica[0])]  # no replacements needed for 0-th replica (master)
        for i in range(1, len(devices)):
            inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
        mse_components = torch.nn.parallel.parallel_apply(funcs_by_replica, inputs_by_replica, devices=devices)
        return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_components)).sum()

    def _replace_and_beam_search(self, params_to_replace: nn.ParameterDict, selection: slice, **kwargs) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then run beam search"""
        dtype = self.quantized_weight.codebooks.dtype
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        out_channel_selection = slice(
            selection.start * self.quantized_weight.out_group_size,
            selection.stop * self.quantized_weight.out_group_size,
        )
        reference_weight = self.layer.weight.detach()[out_channel_selection].to(dtype)
        return self.quantized_weight.beam_search_update_codes_(
            XTX=self.XTX.to(dtype), reference_weight=reference_weight, selection=selection, **kwargs
        ).clone()

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        devices: Sequence[torch.device],
        replicas: Sequence[AQEngine],
        parameters_to_replicate: nn.ParameterDict,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Update quantized_weight codes in-place via beam search"""
        if len(devices) == 1:  # single device
            assert replicas is None
            dtype = self.quantized_weight.codebooks.dtype
            self.quantized_weight.beam_search_update_codes_(
                XTX=self.XTX.to(dtype),
                reference_weight=self.layer.weight.detach().to(dtype),
                dim_rng=random.Random(seed),
                **kwargs,
            )
        else:
            assert replicas[0] is self
            replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices)
            num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
            shard_size = (num_output_groups - 1) // len(devices) + 1
            active_slices_by_replica = [
                slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
            ]

            funcs_by_replica = [replica._replace_and_beam_search for replica in replicas]
            inputs_by_replica = [(dict(), active_slices_by_replica[0])]
            for i in range(1, len(devices)):
                inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
            kwargs_by_replica = [dict(kwargs, dim_rng=random.Random(seed)) for _ in range(len(devices))]
            new_code_parts_by_replica = torch.nn.parallel.parallel_apply(
                funcs_by_replica, inputs_by_replica, kwargs_by_replica, devices=devices
            )
            # gather all code parts and assign them to each replica
            for device, replica in zip(devices, replicas):
                replica.quantized_weight.set_codes(Gather.apply(device, 0, *new_code_parts_by_replica))


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)
