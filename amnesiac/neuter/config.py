"""Configuration constants for summary neutering."""

MODEL_N = "deepseek/deepseek-v4-flash"
MODEL_J1 = "qwen/qwen3.5-397b-a17b"
MODEL_J2 = "anthropic/claude-haiku-4.5"

TEMPERATURE_J1 = 0.1
TEMPERATURE_N = 0.3
J2_TEMPERATURE = 1.0

N_MAX = 3
Y_FLOOR = 0.70
ROLLBACK_ON_HARD_RESIDUAL = False
J2_N_SAMPLES = 5
JUDGE_BLIND_THRESHOLD = 0.4

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_TIMEOUT_SECONDS = 600.0
J2_SEMAPHORE_LIMIT = 5

