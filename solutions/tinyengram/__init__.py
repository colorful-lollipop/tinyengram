"""TinyEngram: conditional memory from scratch.

Public API is intentionally flat so the labs can import what they need:

    from tinyengram import EngramConfig, Engram, NgramHashMapping, ...
"""

from .allocation import (
    Allocation,
    allocate,
    argmin_quadratic,
    engram_parameter_count,
    iso_flops,
    power_law_fit,
    power_law_predict,
    quadratic_fit,
    u_shaped_loss,
)
from .analysis import binary_entropy, gate_stats, linear_cka, logit_lens_kl, soft_alignment
from .config import EngramConfig, TinyLMConfig
from .conv import ShortConv
from .data import EntityCorpus, ToyTokenizer, pad_batch
from .embedding import EngramMemory, MultiHeadEmbedding
from .engram import Engram, build_engram_mapping, build_engram_modules
from .gating import RMSNorm, context_gate, gated_value, rms_norm, signed_sqrt
from .hashing import NgramHashMapping, is_prime, make_odd_multipliers, next_primes
from .inference import generate, naive_generate, sample_next
from .systems import (
    LRUCache,
    all_to_all_counts,
    OffloadResult,
    estimate_transfer_ms,
    prefetch_plan,
    shard_gather,
    simulate_cache,
    simulate_offload,
    zipf_probs,
)
from .tokenizer import CompressedTokenizer, normalize_token_text
from .train import TrainConfig, Trainer, evaluate, make_param_groups, train_step
from .transformer import (
    Block,
    CausalSelfAttention,
    RotaryEmbedding,
    SwiGLU,
    TinyLM,
    apply_rope,
    count_parameters,
)

__all__ = [
    "Allocation",
    "Block",
    "CausalSelfAttention",
    "CompressedTokenizer",
    "Engram",
    "EngramConfig",
    "EngramMemory",
    "EntityCorpus",
    "LRUCache",
    "MultiHeadEmbedding",
    "NgramHashMapping",
    "OffloadResult",
    "RMSNorm",
    "RotaryEmbedding",
    "ShortConv",
    "SwiGLU",
    "TinyLM",
    "TinyLMConfig",
    "ToyTokenizer",
    "TrainConfig",
    "Trainer",
    "all_to_all_counts",
    "allocate",
    "apply_rope",
    "argmin_quadratic",
    "binary_entropy",
    "build_engram_mapping",
    "build_engram_modules",
    "context_gate",
    "count_parameters",
    "engram_parameter_count",
    "estimate_transfer_ms",
    "evaluate",
    "gated_value",
    "gate_stats",
    "generate",
    "iso_flops",
    "is_prime",
    "linear_cka",
    "logit_lens_kl",
    "make_odd_multipliers",
    "make_param_groups",
    "naive_generate",
    "next_primes",
    "normalize_token_text",
    "pad_batch",
    "power_law_fit",
    "power_law_predict",
    "prefetch_plan",
    "quadratic_fit",
    "rms_norm",
    "sample_next",
    "shard_gather",
    "signed_sqrt",
    "simulate_cache",
    "simulate_offload",
    "soft_alignment",
    "train_step",
    "u_shaped_loss",
    "zipf_probs",
]
