"""Montagem do aviso de limite e cálculo do teto rígido."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

from roteador.config import Config
from roteador.tipos import Faixa


def montar_prompt(prompt: str, x: int) -> str:
    """Acrescenta ao final do prompt o aviso de gasto máximo aproximado."""
    _validar_x(x)
    prefixo = f"{prompt} " if prompt else ""
    return f"{prefixo}Limite seus gastos a cerca de {x} tokens."


def teto_rigido(x: int, cfg: Config) -> int:
    """Calcula o teto rígido como ceil(fator_teto × X)."""
    _validar_x(x)
    return ceil(cfg.fator_teto * x)


def limite_por_faixa(faixa: Faixa, cortes: Sequence[float]) -> int | None:
    """Devolve X para a faixa média e None para as demais faixas."""
    if faixa != Faixa.MEDIA:
        return None
    if len(cortes) < 2:
        raise ValueError("São necessários dois cortes para definir a faixa média.")

    x = int(cortes[1])
    _validar_x(x)
    return x


def _validar_x(x: int) -> None:
    if x <= 0:
        raise ValueError("O limite X deve ser maior que zero.")
