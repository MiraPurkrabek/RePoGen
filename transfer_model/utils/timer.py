# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import time
import numpy as np
import torch

from loguru import logger


class Timer(object):
    def __init__(self, name='', sync=False):
        super(Timer, self).__init__()
        self.elapsed = []
        self.name = name
        self.sync = sync

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        if self.sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        self.elapsed.append(elapsed)
        logger.info(f'[{self.name}]: {np.mean(self.elapsed):.3f}')


def timer_decorator(sync=False, name=''):
    def wrapper(method):
        elapsed = []

        def timed(*args, **kw):
            if sync:
                torch.cuda.synchronize()
            ts = time.perf_counter()
            result = method(*args, **kw)
            if sync:
                torch.cuda.synchronize()
            te = time.perf_counter()
            elapsed.append(te - ts)
            logger.info(f'[{name}]: {np.mean(elapsed):.3f}')
            return result
        return timed
    return wrapper
