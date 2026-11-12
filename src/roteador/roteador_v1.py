"""Roteador v1: decide apenas pela contagem de tokens."""

from __future__ import annotations

from roteador.config import Config
from roteador.tipos import Decisao, Papel, Tokenizador


class RoteadorV1:
    """Decide entre barato e forte por tokens da pergunta e do contexto."""

    def __init__(self, tokenizador: Tokenizador, cfg: Config) -> None:
        self.tokenizador = tokenizador
        self.cfg = cfg

    def decidir(self, pergunta: str, contexto: str = "") -> Decisao:
        tokens_pergunta = self.tokenizador.contar(pergunta)
        tokens_contexto = self.tokenizador.contar(contexto)
        tokens_totais = tokens_pergunta + tokens_contexto

        if tokens_totais > self.cfg.janela_contexto_barato:
            return Decisao(
                papel=Papel.FORTE,
                motivo="contexto não cabe no barato",
                limite_tokens=self.cfg.x_tokens_forte_limitado,
            )

        if tokens_pergunta > self.cfg.limiar_tokens_pergunta:
            return Decisao(
                papel=Papel.FORTE,
                motivo="pergunta acima do limiar",
                limite_tokens=self.cfg.x_tokens_forte_limitado,
            )

        return Decisao(
            papel=Papel.BARATO,
            motivo="pergunta dentro do limiar do barato",
        )
