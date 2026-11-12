"""Contrato comum e erros dos adapters de modelo."""

from __future__ import annotations

from dataclasses import dataclass

from roteador.tipos import Adapter

__all__ = [
    "Adapter",
    "ChamadaAdapter",
    "ErroAdapter",
    "inteiro_ou_zero",
    "status_transitorio",
]


class ErroAdapter(Exception):
    """Falha ao chamar ou interpretar a resposta de um provedor de modelo.

    `transitorio` indica se vale a pena repetir a chamada. Falha de rede,
    HTTP 5xx, 408, 425, 429 e JSON invalido sao transitorios. Erros HTTP
    4xx permanentes nao sao.
    """

    def __init__(self, mensagem: str, *, transitorio: bool = True) -> None:
        super().__init__(mensagem)
        self.transitorio = transitorio


@dataclass(frozen=True)
class ChamadaAdapter:
    """Registro de uma chamada recebida por um adapter."""

    prompt: str
    limite_tokens: int | None


def inteiro_ou_zero(valor: object) -> int:
    """Converte contagens do provedor em inteiro não negativo; ausente vira 0."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return 0
    return max(0, int(valor))


def status_transitorio(status: int) -> bool:
    """Diz se um status HTTP justifica nova tentativa."""
    return status >= 500 or status in (408, 425, 429)
