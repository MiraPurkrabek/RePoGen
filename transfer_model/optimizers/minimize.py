# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

from typing import List, Union, Callable, Optional, Dict
import torch
from loguru import logger
from tqdm import tqdm

from transfer_model.utils import (
    from_torch, Tensor, Array, rel_change)


def minimize(
    optimizer: torch.optim,
    closure,
    params: List[Tensor],
    summary_closure: Optional[Callable[[], Dict[str, float]]] = None,
    maxiters=100,
    ftol=-1.0,
    gtol=1e-9,
    interactive=True,
    summary_steps=10,
    **kwargs
):
    ''' Helper function for running an optimization process
        Args:
            - optimizer: The PyTorch optimizer object
            - closure: The function used to calculate the gradients
            - params: a list containing the parameters that will be optimized
        Keyword arguments:
            - maxiters (100): The maximum number of iterations for the
              optimizer
            - ftol: The tolerance for the relative change in the loss
              function.
              If it is lower than this value, then the process stops
            - gtol: The tolerance for the maximum change in the gradient.
              If the maximum absolute values of the all gradient tensors
              are less than this, then the process will stop.
    '''
    prev_loss = None
    for n in tqdm(range(maxiters), desc='Fitting iterations'):
        loss = optimizer.step(closure)

        if n > 0 and prev_loss is not None and ftol > 0:
            loss_rel_change = rel_change(prev_loss, loss.item())

            if loss_rel_change <= ftol:
                prev_loss = loss.item()
                break

        if (all([var.grad.view(-1).abs().max().item() < gtol
                 for var in params if var.grad is not None]) and gtol > 0):
            prev_loss = loss.item()
            break

        if interactive and n % summary_steps == 0:
            logger.info(f'[{n:05d}] Loss: {loss.item():.4f}')
            if summary_closure is not None:
                summaries = summary_closure()
                for key, val in summaries.items():
                    logger.info(f'[{n:05d}] {key}: {val:.4f}')

        prev_loss = loss.item()

    # Save the final step
    if interactive:
        logger.info(f'[{n + 1:05d}] Loss: {loss.item():.4f}')
        if summary_closure is not None:
            summaries = summary_closure()
            for key, val in summaries.items():
                logger.info(f'[{n + 1:05d}] {key}: {val:.4f}')

    return prev_loss
