"""Geração da curva de custo × qualidade em CSV e Markdown."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from roteador.avaliacao.politicas import ResultadoPolitica

COLUNAS_CURVA = (
    "politica",
    "qualidade_media",
    "custo_estimado_usd",
    "custo_medio_usd",
    "qualidade_relativa_pct",
    "custo_relativo_pct",
    "economia_pct",
    "linhas",
)


def linhas_curva(
    resultados: Sequence[ResultadoPolitica],
) -> list[dict[str, Any]]:
    """Converte os resultados disponíveis nas linhas da curva."""
    disponiveis = [resultado for resultado in resultados if resultado.disponivel]
    if not disponiveis:
        raise ValueError("Nenhuma política disponível para gerar a curva.")

    return [
        {
            "politica": resultado.nome,
            "qualidade_media": resultado.qualidade_media,
            "custo_estimado_usd": resultado.custo_estimado_usd,
            "custo_medio_usd": resultado.custo_medio_usd,
            "qualidade_relativa_pct": resultado.qualidade_relativa_pct,
            "custo_relativo_pct": resultado.custo_relativo_pct,
            "economia_pct": resultado.economia_pct,
            "linhas": resultado.linhas,
        }
        for resultado in disponiveis
    ]


def salvar_curva(
    resultados: Sequence[ResultadoPolitica], caminho: str | Path
) -> list[dict[str, Any]]:
    """Grava a curva.csv e devolve as linhas gravadas."""
    linhas = linhas_curva(resultados)
    arquivo = Path(caminho)
    arquivo.parent.mkdir(parents=True, exist_ok=True)

    with arquivo.open("w", encoding="utf-8", newline="") as aberto:
        escritor = csv.DictWriter(aberto, fieldnames=COLUNAS_CURVA)
        escritor.writeheader()
        escritor.writerows(linhas)

    return linhas


def frase_resultado(resultado: ResultadoPolitica) -> str:
    """Monta a frase principal da curva para uma política."""
    if resultado.economia_pct >= 0:
        gasto = f"gastando {resultado.economia_pct:.1f}% menos"
    else:
        gasto = f"gastando {abs(resultado.economia_pct):.1f}% mais"

    return (
        f"Mantive {resultado.qualidade_relativa_pct:.1f}% da qualidade "
        f"do forte {gasto}."
    )


def tabela_markdown(resultados: Sequence[ResultadoPolitica]) -> str:
    """Monta a tabela em Markdown com a frase de cada política."""
    disponiveis = [resultado for resultado in resultados if resultado.disponivel]
    if not disponiveis:
        raise ValueError("Nenhuma política disponível para gerar a tabela.")

    linhas = [
        "| Política | Qualidade média | Custo estimado (USD) | "
        "Qualidade vs. forte | Custo vs. forte | Economia | Resultado |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for resultado in disponiveis:
        linhas.append(
            "| "
            f"{resultado.nome} | "
            f"{resultado.qualidade_media:.1%} | "
            f"{resultado.custo_estimado_usd:.6f} | "
            f"{resultado.qualidade_relativa_pct:.1f}% | "
            f"{resultado.custo_relativo_pct:.1f}% | "
            f"{resultado.economia_pct:.1f}% | "
            f"{frase_resultado(resultado)} |"
        )
    return "\n".join(linhas)


def gerar_curva(
    resultados: Sequence[ResultadoPolitica], caminho: str | Path
) -> str:
    """Grava a curva.csv e devolve a tabela em Markdown."""
    salvar_curva(resultados, caminho)
    return tabela_markdown(resultados)
