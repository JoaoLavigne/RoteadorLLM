"""Embeddings usados pelos classificadores do roteador v2."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


class ErroEmbedder(Exception):
    """Falha ao gerar ou interpretar embeddings."""


@runtime_checkable
class Embedder(Protocol):
    """Contrato de qualquer gerador de embeddings."""

    nome: str

    def embutir(self, textos: Sequence[str]) -> np.ndarray: ...


class EmbedderFalso:
    """Gera vetores determinísticos a partir do hash do texto."""

    def __init__(self, dimensao: int = 8) -> None:
        if dimensao <= 0:
            raise ValueError("A dimensão do embedder deve ser maior que zero.")
        self.dimensao = dimensao
        self.nome = f"falso:{dimensao}"

    def embutir(self, textos: Sequence[str]) -> np.ndarray:
        if not textos:
            return np.empty((0, self.dimensao), dtype=float)

        vetores = [self._vetor(texto) for texto in textos]
        return np.asarray(vetores, dtype=float)

    def _vetor(self, texto: str) -> np.ndarray:
        resumo = hashlib.sha256(texto.encode("utf-8")).digest()
        semente = int.from_bytes(resumo, byteorder="big")
        gerador = np.random.default_rng(semente)
        return gerador.random(self.dimensao)


class EmbedderOllama:
    """Gera embeddings com o endpoint /api/embed do Ollama."""

    def __init__(
        self,
        modelo: str = "nomic-embed-text",
        host: str | None = None,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.modelo = modelo
        self.nome = modelo
        self.host = (
            host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self._post = post

    def embutir(self, textos: Sequence[str]) -> np.ndarray:
        lista = list(textos)
        if not lista:
            return np.empty((0, 0), dtype=float)

        payload = {"model": self.modelo, "input": lista}
        try:
            resposta_http = self._post_fn()(
                f"{self.host}/api/embed", json=payload
            )
        except Exception as exc:
            raise ErroEmbedder(
                f"Falha de rede ao gerar embeddings no Ollama em {self.host}: {exc}"
            ) from exc

        status = getattr(resposta_http, "status_code", 200)
        if status >= 400:
            corpo = getattr(resposta_http, "text", "")
            raise ErroEmbedder(
                f"Ollama respondeu HTTP {status} ao gerar embeddings: "
                f"{str(corpo)[:200]}"
            )

        try:
            dados = resposta_http.json()
        except Exception as exc:
            raise ErroEmbedder(
                f"Resposta de embeddings do Ollama não é um JSON válido: {exc}"
            ) from exc

        if not isinstance(dados, dict):
            raise ErroEmbedder(
                "Resposta de embeddings do Ollama não é um objeto JSON."
            )

        vetores = dados.get("embeddings")
        if not isinstance(vetores, list) or len(vetores) != len(lista):
            raise ErroEmbedder(
                "Resposta de embeddings do Ollama não tem um vetor por texto."
            )

        try:
            matriz = np.asarray(vetores, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ErroEmbedder(
                f"Embeddings do Ollama não formam uma matriz numérica: {exc}"
            ) from exc

        if matriz.ndim != 2:
            raise ErroEmbedder(
                "Embeddings do Ollama não formam uma matriz bidimensional."
            )
        return matriz

    def _post_fn(self) -> Callable[..., Any]:
        if self._post is not None:
            return self._post
        import requests

        return requests.post
