"""
SMTT Heuristic Discovery - Preflight Verification
=================================================
Performs mandatory end-to-end sanity checks before launching discovery:
1. Verifies Python scientific package dependencies (numpy, scipy, numba, pyyaml, requests).
2. Verifies LLM backend connectivity (Ollama at http://localhost:11434).
3. Verifies target model availability (default: qwen2.5-coder:7b-instruct-q4_K_M).
4. Executes exactly ONE real LLM inference call.
5. Verifies AST sandbox scanning, execution safety, and evaluation on an SMTT instance.
6. Verifies cryptographic provenance logging (SHA-256 prompt & response hashing).

Fails cleanly with actionable error instructions if any check fails.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_packages():
    print("[1/6] Checking Python package dependencies...")
    required = {
        "numpy": "1.24.0",
        "scipy": "1.10.0",
        "numba": "0.58.0",
        "yaml": "6.0",
        "requests": "2.28.0",
    }
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  [FAIL] Missing required packages: {missing}")
        print("  Install with: pip install -r requirements.txt")
        return False
    print("  [PASS] All required packages imported successfully.")
    return True


def check_config():
    print("[2/6] Loading experiment configuration...")
    import yaml
    config_file = PROJECT_ROOT / "config.yaml"
    if not config_file.exists():
        example_file = PROJECT_ROOT / "config.example.yaml"
        if example_file.exists():
            print("  [NOTE] 'config.yaml' not found. Using defaults from 'config.example.yaml'.")
            with open(example_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            print("  [FAIL] Neither 'config.yaml' nor 'config.example.yaml' found.")
            return None
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("  [PASS] Configuration loaded.")
    return config


def check_llm_backend(config):
    print("[3/6] Checking LLM backend connectivity and target model...")
    from src.llm_client import create_llm_client

    try:
        client = create_llm_client(config)
    except Exception as e:
        print(f"  [FAIL] Could not initialize LLM client: {e}")
        return None

    status = client.health_check()
    if not status.get("online"):
        print(f"  [FAIL] LLM backend is offline: {status.get('error')}")
        print("\n  Troubleshooting for Ollama:")
        print("    1. Start the Ollama server: `ollama serve`")
        print(f"    2. Pull target model:     `ollama pull {client.model}`")
        print("    3. If running remotely, check base_url in config.yaml")
        return None

    print(f"  [PASS] Backend online at {getattr(client, 'base_url', 'endpoint')}.")
    target_model = status.get("target_model")
    if not status.get("target_model_present"):
        avail = status.get("models_available", [])
        print(f"  [FAIL] Target model '{target_model}' not found in available models: {avail}")
        print(f"  To install model, run: `ollama pull {target_model}`")
        return None

    print(f"  [PASS] Target model '{target_model}' is available.")
    return client


def test_real_inference(client):
    print("[4/6] Executing test real LLM inference call...")
    test_prompt = (
        "You are an expert algorithm designer. Return a minimal Python function:\n"
        "```python\n"
        "def custom_heuristic(p, d):\n"
        "    import numpy as np\n"
        "    return np.argsort(d)\n"
        "```"
    )

    try:
        gen = client.generate(test_prompt)
    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        return None

    print(f"  [PASS] Real inference succeeded in {gen.inference_time_ms:.1f}ms ({gen.generated_tokens} tokens).")
    print(f"  Prompt SHA-256:   {gen.prompt_hash[:16]}...")
    print(f"  Response SHA-256: {gen.response_hash[:16]}...")
    return gen


def test_sandbox_and_eval(gen_result):
    print("[5/6] Testing AST scanner and sandbox execution...")
    from src.evaluator import safe_execute_candidate, evaluate_on_dataset
    from src.instance_generator import generate_single_instance
    import numpy as np

    code = gen_result.extracted_code
    is_safe, fn, msg = safe_execute_candidate(code)
    if not is_safe or fn is None:
        print(f"  [FAIL] Sandbox execution check failed: {msg}")
        return False
    print(f"  [PASS] AST scan passed and candidate function compiled.")

    # Test evaluation on 1 instance
    rng = np.random.default_rng(42)
    inst = generate_single_instance(n_jobs=20, rdd=0.6, tf=0.6, rng=rng)
    is_valid, mean_tard, elapsed_ms = evaluate_on_dataset(fn, [inst], timeout_s=2.0)

    if not is_valid:
        print(f"  [FAIL] Candidate did not produce a valid permutation or exceeded timeout.")
        return False

    print(f"  [PASS] Instance evaluated successfully (Tardiness: {mean_tard:.0f}, Time: {elapsed_ms:.2f}ms).")
    return True


def test_provenance_logging(gen_result):
    print("[6/6] Verifying provenance logging...")
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    test_log = logs_dir / "preflight_provenance.jsonl"

    record = {
        "status": "PREFLIGHT_VERIFIED",
        "model": gen_result.model,
        "backend": gen_result.backend,
        "timestamp_utc": gen_result.timestamp_utc,
        "prompt_hash": gen_result.prompt_hash,
        "response_hash": gen_result.response_hash,
        "generated_tokens": gen_result.generated_tokens,
        "inference_time_ms": gen_result.inference_time_ms,
    }

    with open(test_log, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"  [PASS] Cryptographic provenance recorded to {test_log}.")
    return True


def main():
    print("=" * 70)
    print("SMTT Heuristic Discovery Preflight Diagnostic")
    print("=" * 70)

    if not check_packages():
        sys.exit(1)

    config = check_config()
    if config is None:
        sys.exit(1)

    client = check_llm_backend(config)
    if client is None:
        print("\nPreflight check aborted due to backend unavailability.")
        sys.exit(1)

    gen = test_real_inference(client)
    if gen is None:
        sys.exit(1)

    if not test_sandbox_and_eval(gen):
        sys.exit(1)

    if not test_provenance_logging(gen):
        sys.exit(1)

    print("\n" + "=" * 70)
    print("PREFLIGHT PASSED: All dependencies, backend, sandbox, and logging verified.")
    print("You are ready to run: python run_pilot.py")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
