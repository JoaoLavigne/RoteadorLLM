"""Tokenizadores real (tiktoken) e aproximado, com reserva automática."""

from __future__ import annotations

from math import ceil
from typing import Any

from roteador.config import Config


class ErroTokenizador(Exception):
    """Falha ao carregar ou usar o tokenizador real."""


class TokenizadorAproximado:
    """Conta tokens aproximando uma quantidade fixa de caracteres por token."""

    nome = "aproximado"

    def __init__(self, divisor: int = 4) -> None:
        if divisor <= 0:
            raise ValueError("O divisor do tokenizador aproximado deve ser maior que zero.")
        self.divisor = divisor

    def contar(self, texto: str) -> int:
        if not texto:
            return 0
        return ceil(len(texto) / self.divisor)


class TokenizadorReal:
    """Conta tokens com o tiktoken, carregando a biblioteca só no primeiro uso."""

    nome = "real"

    def __init__(self, especificacao: str) -> None:
        provedor, separador, codificacao = especificacao.partition(":")
        if provedor != "tiktoken" or not separador or not codificacao:
            raise ErroTokenizador(
                f"Especificação de tokenizador inválida: {especificacao!r}. "
                "Use o formato 'tiktoken:<codificacao>', por exemplo 'tiktoken:o200k_base'."
            )
        self.codificacao = codificacao
        self._codificador: Any = None

    def _carregar(self) -> None:
        if self._codificador is not None:
            return

        try:
            import tiktoken
        except ImportError as exc:
            raise ErroTokenizador(
                "A biblioteca 'tiktoken' não está instalada. "
                "Instale com 'pip install tiktoken'."
            ) from exc

        try:
            self._codificador = tiktoken.get_encoding(self.codificacao)
        except Exception as exc:
            raise ErroTokenizador(
                f"Não foi possível obter a codificação {self.codificacao!r} do tiktoken: {exc}"
            ) from exc

    def contar(self, texto: str) -> int:
        self._carregar()
        return len(self._codificador.encode(texto))


class TokenizadorComReserva:
    """Usa o tokenizador real e cai para o aproximado se a carga falhar."""

    def __init__(self, principal: Any, reserva: Any) -> None:
        self.principal = principal
        self.reserva = reserva
        self._usado: str | None = None

    @property
    def nome(self) -> str:
        if self._usado is not None:
            return self._usado
        return getattr(self.principal, "nome", "real")

    @property
    def tokenizador_usado(self) -> str:
        return self.nome

    def contar(self, texto: str) -> int:
        if self._usado == getattr(self.reserva, "nome", "aproximado"):
            return self.reserva.contar(texto)

        try:
            total = self.principal.contar(texto)
        except ErroTokenizador:
            self._usado = getattr(self.reserva, "nome", "aproximado")
            return self.reserva.contar(texto)

        self._usado = getattr(self.principal, "nome", "real")
        return total


def criar_tokenizador(cfg: Config) -> TokenizadorComReserva:
    """Monta o tokenizador real com o aproximado como reserva."""
    return TokenizadorComReserva(
        TokenizadorReal(cfg.tokenizador),
        TokenizadorAproximado(cfg.divisor_aproximado),
    )
