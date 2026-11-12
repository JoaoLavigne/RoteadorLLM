"""Adapter do modelo forte, servido pelo Gemini."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import requests

from roteador.adapters.base import ErroAdapter, inteiro_ou_zero, status_transitorio
from roteador.tipos import MotivoParada, RespostaModelo

_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class AdapterGemini:
    """Chama o generateContent do Gemini e normaliza a resposta."""

    def __init__(
        self,
        modelo: str | None = None,
        api_key: str | None = None,
        post: Callable[..., Any] | None = None,
        url_base: str = _URL_BASE,
    ) -> None:
        self.modelo = modelo or os.environ.get("GEMINI_MODEL")
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

        faltando = []
        if not self.modelo:
            faltando.append("GEMINI_MODEL")
        if not self._api_key:
            faltando.append("GEMINI_API_KEY")
        if faltando:
            raise ErroAdapter(
                "Configuração do Gemini ausente: "
                f"defina {', '.join(faltando)} no ambiente."
            )

        self.nome = self.modelo
        self._url_base = url_base.rstrip("/")
        self._post = post or requests.post

    def gerar(self, prompt: str, limite_tokens: int | None = None) -> RespostaModelo:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        if limite_tokens is not None:
            payload["generationConfig"] = {"maxOutputTokens": limite_tokens}

        url = f"{self._url_base}/{self.modelo}:generateContent"

        inicio = time.perf_counter()
        try:
            resposta_http = self._post(url, params={"key": self._api_key}, json=payload)
        except Exception as exc:
            raise ErroAdapter(f"Falha de rede ao chamar o Gemini: {exc}") from exc
        tempo_ms = (time.perf_counter() - inicio) * 1000

        status = getattr(resposta_http, "status_code", 200)
        if status >= 400:
            corpo = getattr(resposta_http, "text", "")
            raise ErroAdapter(
                f"Gemini respondeu HTTP {status}: {str(corpo)[:200]}",
                transitorio=status_transitorio(status),
            )

        try:
            dados = resposta_http.json()
        except Exception as exc:
            raise ErroAdapter(f"Resposta do Gemini não é um JSON válido: {exc}") from exc

        if not isinstance(dados, dict):
            raise ErroAdapter("Resposta do Gemini não é um objeto JSON.")

        return self._converter(dados, tempo_ms)

    def _converter(self, dados: dict[str, Any], tempo_ms: float) -> RespostaModelo:
        candidato = _primeiro_candidato(dados)
        texto = _texto_do_candidato(candidato)
        motivo = _normalizar_motivo(texto, candidato.get("finishReason"))

        uso = dados.get("usageMetadata")
        if not isinstance(uso, dict):
            uso = {}

        tokens_raciocinio = None
        if "thoughtsTokenCount" in uso:
            tokens_raciocinio = inteiro_ou_zero(uso.get("thoughtsTokenCount"))

        return RespostaModelo(
            texto=texto,
            tokens_entrada=inteiro_ou_zero(uso.get("promptTokenCount")),
            tokens_saida=inteiro_ou_zero(uso.get("candidatesTokenCount")),
            tokens_raciocinio=tokens_raciocinio,
            motivo_parada=motivo,
            tempo_ms=tempo_ms,
            modelo=self.modelo,
        )


def _primeiro_candidato(dados: dict[str, Any]) -> dict[str, Any]:
    candidatos = dados.get("candidates")
    if isinstance(candidatos, list) and candidatos and isinstance(candidatos[0], dict):
        return candidatos[0]
    return {}


def _texto_do_candidato(candidato: dict[str, Any]) -> str:
    conteudo = candidato.get("content")
    if not isinstance(conteudo, dict):
        return ""

    partes = conteudo.get("parts")
    if not isinstance(partes, list):
        return ""

    textos = []
    for parte in partes:
        if isinstance(parte, dict) and isinstance(parte.get("text"), str):
            textos.append(parte["text"])
    return "".join(textos)


def _normalizar_motivo(texto: str, finish_reason: object) -> MotivoParada:
    if not texto:
        return MotivoParada.VAZIO
    if finish_reason == "MAX_TOKENS":
        return MotivoParada.TRUNCADO
    return MotivoParada.COMPLETO
