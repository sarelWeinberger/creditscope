"""SGLang inference server configuration for Qwen3.5-35B-A3B-FP8."""

import os

MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3.5-35B-A3B-FP8")
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "32768"))
TP_SIZE = int(os.getenv("TP_SIZE", "1"))
PORT = int(os.getenv("INFERENCE_PORT", "8000"))
MEM_FRACTION_STATIC = float(os.getenv("MEM_FRACTION_STATIC", "0.85"))
REASONING_PARSER = "qwen3"
TOOL_CALL_PARSER = "qwen3_coder"

# MoE architecture details for Qwen3.5-35B-A3B
NUM_EXPERTS = 64
TOP_K_EXPERTS = 4
MOE_TRACE_BUFFER_SIZE = int(os.getenv("MOE_TRACE_BUFFER_SIZE", "100"))

# Prometheus metrics
METRICS_ENABLED = os.getenv("ENABLE_METRICS", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
