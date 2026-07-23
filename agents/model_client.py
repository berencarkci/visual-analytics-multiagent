"""Model clients for the Visual Analytics Assistant.

One swappable interface so the same pipeline runs the prompt-only base model and the SFT/DPO checkpoints later. 
HFClient use local Hugging Face weights (mps > cuda > cpu, picked automatically).
MockClient enables pipeline tests without loading any model.
"""

from __future__ import annotations

from typing import Protocol

# Client interface:
class ModelClient(Protocol):
    """Anything with generate(messages) -> str is a valid client"""

    def generate(self, messages: list[dict]) -> str: ...
#################################


# Hugging Face transformers client:
class HFClient:
    """Local HF chat model, lazy loads on first generate call so we dont download the model on mockClient
    temperature=0.0 -> greedy decoding (deterministic, always the same answer for the same question), the default forbenchmark runs so results are reproducible. 
    Candidate generation for preference pairs (B4) will have temperature > 0.

    Runs the 3B model locally and on Colab for the final benchmark runs.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", temperature: float = 0.0, max_new_tokens: int = 300, adapter: str | None = None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.adapter = adapter # LoRA adapter: local path or Hub id
        self._model = None
        self._tokenizer = None
        self.device = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        elif torch.cuda.is_available():
            self.device, dtype = "cuda", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device)
        if self.adapter: # SFT/DPO checkpoint
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter)
            self._model = self._model.merge_and_unload() # fold LoRA in: same speed as base
        self._model.eval()

    def generate(self, messages: list[dict]) -> str:
        import torch

        if self._model is None:
            self._load()

        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self._model.generate(**inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
#################################


# Mock client for tests:
class MockClient:
    """Returns queued responses in order, lets us test the retry flow without model, for quick tests"""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.call_count = 0

    def generate(self, messages: list[dict]) -> str:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]
#################################