"""Adapter do modelo barato, servido pelo Ollama."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import requests

from roteador.adapters.base import ErroAdapter, inteiro_ou_zero, status_transitorio
from roteador.tipos import MotivoParada, RespostaModelo


class AdapterOllama:
    """Chama o endpoint /api/chat do Ollama e normaliza a resposta."""

    def __init__(
        self,
        modelo: str = "llama3.2:3b",
        host: str | None = None,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.modelo = modelo
        self.nome = modelo
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self._post = post or requests.post

    def gerar(self, prompt: str, limite_tokens: int | None = None) -> RespostaModelo:
        payload: dict[str, Any] = {
            "model": self.modelo,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if limite_tokens is not None:
            payload["options"] = {"num_predict": limite_tokens}

        inicio = time.perf_counter()
        try:
            resposta_http = self._post(f"{self.host}/api/chat", json=payload)
        except Exception as exc:
            raise ErroAdapter(f"Falha de rede ao chamar o Ollama em {self.host}: {exc}") from exc
        tempo_ms = (time.perf_counter() - inicio) * 1000

        status = getattr(resposta_http, "status_code", 200)
        if status >= 400:
            corpo = getattr(resposta_http, "text", "")
            raise ErroAdapter(
                f"Ollama respondeu HTTP {status}: {str(corpo)[:200]}",
                transitorio=status_transitorio(status),
            )

        try:
            dados = resposta_http.json()
        except Exception as exc:
            raise ErroAdapter(f"Resposta do Ollama não é um JSON válido: {exc}") from exc

        if not isinstance(dados, dict):
            raise ErroAdapter("Resposta do Ollama não é um objeto JSON.")

        return self._converter(dados, tempo_ms)

    def _converter(self, dados: dict[str, Any], tempo_ms: float) -> RespostaModelo:
        mensagem = dados.get("message")
        texto = ""
        if isinstance(mensagem, dict):
            conteudo = mensagem.get("content")
            if isinstance(conteudo, str):
                texto = conteudo

        motivo = _normalizar_motivo(texto, dados.get("done_reason"))

        modelo = dados.get("model")
        if not isinstance(modelo, str) or not modelo:
            modelo = self.modelo

        return RespostaModelo(
            texto=texto,
            tokens_entrada=inteiro_ou_zero(dados.get("prompt_eval_count")),
            tokens_saida=inteiro_ou_zero(dados.get("eval_count")),
            tokens_raciocinio=None,
            motivo_parada=motivo,
            tempo_ms=tempo_ms,
            modelo=modelo,
        )


def _normalizar_motivo(texto: str, done_reason: object) -> MotivoParada:
    if not texto:
        return MotivoParada.VAZIO
    if done_reason == "length":
        return MotivoParada.TRUNCADO
    return MotivoParada.COMPLETO
