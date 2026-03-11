"""
Inference server configuration for Qwen3.5-35B-A3B-FP8 on SGLang.
"""

import os

MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3.5-35B-A3B-FP8")
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "32768"))
TP_SIZE = int(os.getenv("TP_SIZE", "1"))
PORT = int(os.getenv("SGLANG_PORT", "8000"))
MEM_FRACTION_STATIC = float(os.getenv("MEM_FRACTION_STATIC", "0.85"))

REASONING_PARSER = "qwen3"
TOOL_CALL_PARSER = "qwen3_coder"

# MoE architecture details for Qwen3.5-35B-A3B
NUM_EXPERTS = 64
TOP_K_EXPERTS = 4
MOE_LAYER_PATTERN = "mlp.experts"  # Pattern to identify MoE FFN layers

# Thinking budget presets
THINKING_BUDGET_PRESETS = {
    "none": 0,
    "minimal": 128,
    "short": 512,
    "standard": 2048,
    "extended": 8192,
    "deep": 32768,
    "unlimited": -1,
}

# Sampling parameters per thinking mode
SAMPLING_THINKING_ON = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
}

SAMPLING_THINKING_OFF = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
}
