# src/seeding.py
"""Single source of truth for every stochastic component in a run.

A run's randomness is derived from one integer, ``train_seed``:

* python ``random``            — library tie-breaks
* ``numpy.random`` (global)    — augmentation in ``PreprocessedDataset``
* ``torch`` / ``torch.cuda``   — weight init, dropout, cuDNN workspace choices
* DataLoader shuffle order     — from an explicit ``torch.Generator``
* DataLoader worker processes  — reseeded via ``seed_worker``

The worker case is the one that is easy to get wrong. PyTorch reseeds ``torch``
inside each worker (base_seed + worker_id) but leaves numpy's global RNG alone.
With the default fork start method every worker therefore inherits an identical
copy of the parent's numpy state, and that state was never seeded from the
config in the first place — so numpy-based augmentation is both unseeded and
duplicated across workers. ``seed_worker`` closes that hole by deriving each
worker's numpy/random seeds from the torch base seed the parent assigned it.

The split RNG is deliberately NOT handled here: it lives in ``src.splits`` and
is driven by ``split_seed``, which is fixed across the whole study.
"""
from __future__ import annotations

import random

import numpy as np
import torch

# numpy's legacy global seeder accepts [0, 2**32); torch seeds are 64-bit.
_UINT32 = 2 ** 32


def set_seed(seed: int) -> None:
    """Seed every global RNG a run draws from.

    ``torch.manual_seed`` also seeds all CUDA devices, but we call
    ``cuda.manual_seed_all`` explicitly so the guarantee is visible here rather
    than implied by a PyTorch implementation detail.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % _UINT32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by torch
    """DataLoader ``worker_init_fn``: seed numpy and random inside the worker.

    ``torch.initial_seed()`` inside a worker returns the per-worker seed the
    parent derived from the loader's generator, so this is deterministic given
    ``train_seed`` and distinct per worker and per epoch-independent shuffle.
    """
    worker_seed = torch.initial_seed() % _UINT32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """CPU generator for DataLoader shuffling — passed explicitly as ``generator=``.

    Without this the shuffle order comes from the global torch RNG, which is
    also consumed by weight init, so shuffle order silently depends on how many
    random numbers model construction happened to draw.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def capture_rng_state(loader_generator: torch.Generator | None = None) -> dict:
    """Snapshot every RNG so a resumed run continues the same stream.

    Resuming without this restores the weights but not the randomness, which
    silently breaks the seeding guarantee: the augmentation and shuffle streams
    would restart from wherever the fresh process happened to be.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def restore_rng_state(state: dict, loader_generator: torch.Generator | None = None) -> None:
    """Inverse of :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [_as_byte_tensor(s) for s in state["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            # Resumed on a node with a different device count; seed device 0
            # from the saved state and leave the rest at their fresh values.
            torch.cuda.set_rng_state(cuda_states[0])
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(_as_byte_tensor(state["loader_generator"]))


def _as_byte_tensor(x) -> torch.Tensor:
    """torch.load may hand back a tensor on a non-CPU device; RNG states must be CPU uint8."""
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.uint8)
    return t.cpu().to(torch.uint8)
