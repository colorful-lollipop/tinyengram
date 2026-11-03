"""KV pairs plus the bounded Engram activation history needed by decoding.

This supplied container is list-compatible: ``cache[i]`` is still ``(K, V)``.
Do not convert an Engram cache to a plain list: that would lose conv history.
"""

from __future__ import annotations

import torch

__all__ = ["KVCache"]


class KVCache(list):
    def __init__(self, pairs=(), *, conv_windows=None, history_sizes=None, chunk_start=0):
        super().__init__(pairs)
        self.conv_windows = conv_windows if conv_windows is not None else [None] * len(self)
        self.history_sizes = history_sizes if history_sizes is not None else [0] * len(self)
        self.chunk_start = chunk_start
        if len(self.conv_windows) != len(self) or len(self.history_sizes) != len(self):
            raise ValueError("one conv state/history size required per layer")
        self.conv_states = [
            None if window is None else window[:, -size:] if size else window[:, :0]
            for window, size in zip(self.conv_windows, self.history_sizes)
        ]

    @property
    def seq_len(self) -> int:
        return self[0][0].shape[-2] if self else 0

    def crop(self, length: int, *, conv_states: list[torch.Tensor | None] | None = None):
        """Crop KV and restore conv state after a speculative chunk.

        The latest forward retains at most H previous activations plus its new
        chunk, so any prefix of that chunk can be kept. Rolling back several
        separate forwards needs the saved ``conv_states`` at that checkpoint;
        the draft decoder saves these small per-token snapshots, not old KV.
        No input tensors are mutated. State tensors are read-only views.
        """
        if not 0 <= length <= self.seq_len:
            raise ValueError("crop length outside cached sequence")
        if conv_states is not None:
            if len(conv_states) != len(self):
                raise ValueError("checkpoint must contain one conv state per layer")
            windows = conv_states
        else:
            windows = []
            for window, size in zip(self.conv_windows, self.history_sizes):
                if window is None:
                    windows.append(None)
                    continue
                if length < self.chunk_start:
                    raise ValueError("rollback before latest chunk needs a conv checkpoint")
                old_history = window.shape[1] - (self.seq_len - self.chunk_start)
                kept = window[:, :old_history + length - self.chunk_start]
                windows.append(kept[:, -size:] if size else kept[:, :0])
        pairs = [(k[..., :length, :], v[..., :length, :]) for k, v in self]
        return KVCache(pairs, conv_windows=windows, history_sizes=self.history_sizes,
                       chunk_start=length)
