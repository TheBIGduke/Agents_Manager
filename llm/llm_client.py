from __future__ import annotations
import threading
import os
import json
from typing import Optional, Dict, Any
from llama_cpp import Llama

from config.settings import CONTEXT_LLM,THREADS_LLM,N_BATCH_LLM,GPU_LAYERS_LLM,CHAT_FORMAT_LLM,USE_LLM
from config.llm_system_prompt_def import GENERAL_SYSTEM_PROMPT

class LLM:
    def __init__(self, model_path:str, system_prompt: str | None = None):
        self.system = system_prompt or GENERAL_SYSTEM_PROMPT
        self._llm = None
        self._lock = threading.Lock()

        # Defaults sensatos (CPU-only). Ajusta por env si quieres.
        self.model_path = model_path
        self.ctx = CONTEXT_LLM         # contexto razonable
        self.threads = THREADS_LLM
        self.n_batch = N_BATCH_LLM   # 256–512 bien en CPU
        self.n_gpu_layers = GPU_LAYERS_LLM  # 0 si no hay CUDA
        self.chat_format = CHAT_FORMAT_LLM.strip()

    def ensure(self):
        """ Initialize the LLM instance if not already done """
        if self._llm is None and USE_LLM:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            kwargs = dict(
                model_path=self.model_path,
                n_ctx=self.ctx,
                n_threads=self.threads,
                n_batch=self.n_batch,
                n_gpu_layers=self.n_gpu_layers,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            self._llm = Llama(**kwargs)

    def answer_general(self, user_prompt: str) -> str:
        """ Answer a general question with the LLM """

        if not USE_LLM:
            return "El LLM está desactivado en la configuración"

        self.ensure()
        general_system = GENERAL_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": general_system},
            {"role": "user", "content": user_prompt},
        ]
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                top_p=0.9,
                max_tokens=100,
            )
        msg = out["choices"][0]["message"]
        return (msg.get("content") or "").strip() or "No tengo una respuesta."