"""Rótulos derivados do dataset: barato basta, faixas e divisão treino/teste."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from roteador.dados.carregador import LinhaDataset, Nota
from roteador.tipos import Faixa

Cortes = tuple[float, float]

_FAIXAS_EM_ORDEM = (Faixa.CURTA, Faixa.MEDIA, Faixa.LONGA)


def barato_basta(linha: LinhaDataset | Mapping[str, Any]) -> bool:
    """Diz se o barato iguala ou supera o forte na qualidade.

    As notas viram números: correta = 2, parcial = 1, errada = 0. Empate conta
    como "o barato basta".
    """
    nota_barato = _nota_da_linha(linha, "nota_barato")
    nota_forte = _nota_da_linha(linha, "nota_forte")
    return nota_barato >= nota_forte


def calcular_cortes(
    tokens: Sequence[int | float], percentis: Sequence[float] = (1 / 3, 2 / 3)
) -> Cortes:
    """Calcula os dois cortes que separam as faixas curta, média e longa."""
    valores = sorted(float(token) for token in tokens)
    if not valores:
        raise ValueError("Não é possível calcular cortes sem respostas.")

    if len(percentis) != 2:
        raise ValueError("São necessários exatamente dois percentis para as faixas.")

    primeiro, segundo = (float(percentil) for percentil in percentis)
    if not 0 <= primeiro < segundo <= 1:
        raise ValueError(
            "Os percentis devem ser crescentes e estar entre 0 e 1."
        )

    return (_percentil(valores, primeiro), _percentil(valores, segundo))


def faixa_de(
    tokens: int | float, cortes: Cortes, truncada_forte: bool = False
) -> Faixa:
    """Devolve a faixa de tamanho de uma resposta.

    Uma resposta truncada sobe uma faixa, porque o tamanho real é maior ou
    igual ao teto aplicado.
    """
    _validar_cortes(cortes)

    if tokens <= cortes[0]:
        faixa = Faixa.CURTA
    elif tokens <= cortes[1]:
        faixa = Faixa.MEDIA
    else:
        faixa = Faixa.LONGA

    if truncada_forte:
        faixa = _subir_faixa(faixa)
    return faixa


def faixa_da_linha(
    linha: LinhaDataset | Mapping[str, Any], cortes: Cortes
) -> Faixa:
    """Devolve a faixa da resposta do forte, considerando o truncamento."""
    tokens = _campo_da_linha(linha, "tokens_resposta_forte")
    truncada = bool(_campo_da_linha(linha, "truncada_forte", padrao=False))
    return faixa_de(tokens, cortes, truncada_forte=truncada)


def dividir_por_documento(
    linhas: Sequence[LinhaDataset],
    fracao_teste: float = 0.2,
    semente: int = 42,
) -> tuple[list[LinhaDataset], list[LinhaDataset]]:
    """Divide treino e teste sem separar linhas do mesmo documento."""
    if not 0 < fracao_teste < 1:
        raise ValueError("A fração de teste deve estar entre 0 e 1.")

    lista = list(linhas)
    documentos = sorted({_campo_da_linha(linha, "documento") for linha in lista})
    if len(documentos) < 2:
        raise ValueError(
            "A divisão por documento exige pelo menos 2 documentos; "
            f"encontrado(s) {len(documentos)}."
        )

    embaralhador = random.Random(semente)
    embaralhador.shuffle(documentos)

    quantidade_teste = round(len(documentos) * fracao_teste)
    quantidade_teste = max(1, min(len(documentos) - 1, quantidade_teste))
    documentos_teste = set(documentos[:quantidade_teste])

    treino = [
        linha
        for linha in lista
        if _campo_da_linha(linha, "documento") not in documentos_teste
    ]
    teste = [
        linha
        for linha in lista
        if _campo_da_linha(linha, "documento") in documentos_teste
    ]
    return treino, teste


def _percentil(valores: Sequence[float], percentil: float) -> float:
    posicao = percentil * (len(valores) - 1)
    inferior = math.floor(posicao)
    superior = math.ceil(posicao)
    if inferior == superior:
        return valores[inferior]

    peso = posicao - inferior
    return valores[inferior] * (1 - peso) + valores[superior] * peso


def _subir_faixa(faixa: Faixa) -> Faixa:
    indice = _FAIXAS_EM_ORDEM.index(faixa)
    return _FAIXAS_EM_ORDEM[min(indice + 1, len(_FAIXAS_EM_ORDEM) - 1)]


def _validar_cortes(cortes: Sequence[float]) -> None:
    if len(cortes) != 2:
        raise ValueError("São necessários exatamente dois cortes de faixa.")
    if cortes[0] > cortes[1]:
        raise ValueError("Os cortes de faixa devem estar em ordem crescente.")


def _nota_da_linha(
    linha: LinhaDataset | Mapping[str, Any], campo: str
) -> Nota:
    return Nota.de_texto(_campo_da_linha(linha, campo))


def _campo_da_linha(
    linha: LinhaDataset | Mapping[str, Any], campo: str, padrao: Any = None
) -> Any:
    if isinstance(linha, Mapping):
        return linha.get(campo, padrao)
    return getattr(linha, campo, padrao)
