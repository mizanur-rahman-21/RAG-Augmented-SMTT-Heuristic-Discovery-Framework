"""
Unit tests for LLM client utilities and error handling
"""

import unittest
from src.llm_client import extract_code_block, OllamaClient


class TestLLMClient(unittest.TestCase):

    def test_extract_code_block_markdown(self):
        raw_response = """
Here is my improved heuristic:
```python
def custom_heuristic(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.argsort(d)
```
Hope this helps!
"""
        code = extract_code_block(raw_response)
        self.assertIn("def custom_heuristic", code)
        self.assertNotIn("```", code)
        self.assertNotIn("Hope this helps", code)

    def test_extract_code_block_raw(self):
        raw_response = """
def custom_heuristic(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.argsort(d)
"""
        code = extract_code_block(raw_response)
        self.assertEqual(code.strip(), raw_response.strip())

    def test_ollama_offline_error_handling(self):
        # Target an invalid port that is guaranteed not to be running Ollama
        client = OllamaClient(base_url="http://127.0.0.1:59999", timeout_s=1.0)
        status = client.health_check()
        self.assertFalse(status["online"])
        self.assertIn("Failed to connect", status["error"])

        with self.assertRaises(ConnectionError) as ctx:
            client.generate("def custom_heuristic(): pass")
        self.assertIn("Could not connect to Ollama", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
