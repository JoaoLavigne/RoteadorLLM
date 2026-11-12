"""Adapter falso, usado nos testes sem rede."""

from __future__ import annotations

from collections.abc import Iterable

from roteador.adapters.base import ChamadaAdapter, ErroAdapter
from roteador.tipos import RespostaModelo


class AdapterFalso:
    """Devolve respostas programadas em sequência e registra as chamadas."""

    nome = "falso"

    def __init__(
        self,
        respostas: Iterable[RespostaModelo],
        nome: str | None = None,
    ) -> None:
        self._respostas = list(respostas)
        self._indice = 0
        self.chamadas: list[ChamadaAdapter] = []
        if nome is not None:
            self.nome = nome

    def gerar(self, prompt: str, limite_tokens: int | None = None) -> RespostaModelo:
        self.chamadas.append(ChamadaAdapter(prompt, limite_tokens))

        if self._indice >= len(self._respostas):
            raise ErroAdapter(
                "AdapterFalso sem respostas programadas: "
                f"recebeu {len(self.chamadas)} chamada(s), "
                f"mas só tem {len(self._respostas)} resposta(s).",
                transitorio=False,
            )

        resposta = self._respostas[self._indice]
        self._indice += 1
        return resposta
