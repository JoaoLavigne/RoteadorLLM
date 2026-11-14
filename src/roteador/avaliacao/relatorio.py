"""Relatório de alertas de respostas vazias ou truncadas."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from roteador.config import Config
from roteador.tipos import TipoOcorrencia

_OCORRENCIAS_IGNORADAS = {"", TipoOcorrencia.NENHUMA.value}


@dataclass(frozen=True)
class TaxaAlerta:
    """Taxa de ocorrências de um modelo ou de uma faixa."""

    valor: str
    ocorrencias: int
    requisicoes: int
    taxa: float
    alerta: bool


@dataclass(frozen=True)
class RelatorioAlertas:
    """Taxas agrupadas por modelo e por faixa."""

    por_modelo: tuple[TaxaAlerta, ...]
    por_faixa: tuple[TaxaAlerta, ...]

    @property
    def alertas(self) -> tuple[TaxaAlerta, ...]:
        return tuple(
            taxa
            for taxa in (*self.por_modelo, *self.por_faixa)
            if taxa.alerta
        )


def gerar_relatorio_alertas(
    registro: Sequence[Mapping[str, object]],
    ml: Sequence[Mapping[str, object]],
    cfg: Config,
) -> RelatorioAlertas:
    """Calcula ocorrências de ML divididas por requisições do registro.

    As linhas sem modelo ou sem faixa prevista não entram no agrupamento
    correspondente.
    """
    requisicoes_modelo = _contar_requisicoes(registro, "modelo")
    ocorrencias_modelo = _contar_ocorrencias(ml, "modelo")
    requisicoes_faixa = _contar_requisicoes(registro, "faixa_prevista")
    ocorrencias_faixa = _contar_ocorrencias(ml, "faixa_prevista")

    return RelatorioAlertas(
        por_modelo=_montar_taxas(
            requisicoes_modelo, ocorrencias_modelo, cfg.limiar_alerta
        ),
        por_faixa=_montar_taxas(
            requisicoes_faixa, ocorrencias_faixa, cfg.limiar_alerta
        ),
    )


def _contar_requisicoes(
    registro: Sequence[Mapping[str, object]], campo: str
) -> Counter[str]:
    contagem: Counter[str] = Counter()
    for linha in registro:
        valor = _valor(linha, campo)
        if valor:
            contagem[valor] += 1
    return contagem


def _contar_ocorrencias(
    ml: Sequence[Mapping[str, object]], campo: str
) -> Counter[str]:
    contagem: Counter[str] = Counter()
    for linha in ml:
        if _valor(linha, "tipo_ocorrencia") in _OCORRENCIAS_IGNORADAS:
            continue
        valor = _valor(linha, campo)
        if valor:
            contagem[valor] += 1
    return contagem


def _montar_taxas(
    requisicoes: Counter[str],
    ocorrencias: Counter[str],
    limiar: float,
) -> tuple[TaxaAlerta, ...]:
    valores = sorted(set(requisicoes) | set(ocorrencias))
    taxas = []
    for valor in valores:
        total_requisicoes = requisicoes.get(valor, 0)
        total_ocorrencias = ocorrencias.get(valor, 0)
        taxa = (
            total_ocorrencias / total_requisicoes
            if total_requisicoes > 0
            else 0.0
        )
        taxas.append(
            TaxaAlerta(
                valor=valor,
                ocorrencias=total_ocorrencias,
                requisicoes=total_requisicoes,
                taxa=taxa,
                alerta=total_requisicoes > 0 and taxa > limiar,
            )
        )
    return tuple(taxas)


def _valor(linha: Mapping[str, object], campo: str) -> str:
    valor = linha.get(campo, "")
    if valor is None:
        return ""
    return str(valor).strip()


def tabela_markdown_alertas(relatorio: RelatorioAlertas) -> str:
    """Monta uma tabela Markdown com as taxas por modelo e por faixa."""
    linhas = [
        "| Grupo | Valor | Ocorrências | Requisições | Taxa | Alerta |",
        "| --- | --- | ---: | ---: | ---: | :---: |",
    ]
    for grupo, taxas in (
        ("modelo", relatorio.por_modelo),
        ("faixa", relatorio.por_faixa),
    ):
        for taxa in taxas:
            linhas.append(
                "| "
                f"{grupo} | "
                f"{taxa.valor} | "
                f"{taxa.ocorrencias} | "
                f"{taxa.requisicoes} | "
                f"{taxa.taxa:.1%} | "
                f"{'sim' if taxa.alerta else 'não'} |"
            )
    return "\n".join(linhas)
