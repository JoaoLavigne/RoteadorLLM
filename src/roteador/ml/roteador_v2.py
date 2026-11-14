"""Roteador v2: usa os classificadores A e B sobre embedding e tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from roteador.config import Config
from roteador.dados.carregador import LinhaDataset
from roteador.dados.rotulos import barato_basta, calcular_cortes, faixa_da_linha
from roteador.limite import limite_por_faixa
from roteador.ml.classificadores import (
    ModeloRoteador,
    montar_features,
    treinar_a,
    treinar_b,
)
from roteador.ml.embedder import Embedder
from roteador.tipos import Decisao, Faixa, Papel, Tokenizador


class RoteadorV2:
    """Decide pelo classificador A e prevê a faixa com o classificador B."""

    def __init__(
        self,
        embedder: Embedder,
        modelo_a: Any,
        modelo_b: Any,
        tokenizador: Tokenizador,
        cfg: Config,
    ) -> None:
        if not 0 <= cfg.limiar_probabilidade <= 1:
            raise ValueError(
                "O limiar de probabilidade deve estar entre 0 e 1."
            )
        self.embedder = embedder
        self.modelo_a = modelo_a
        self.modelo_b = modelo_b
        self.tokenizador = tokenizador
        self.cfg = cfg

    def decidir(self, pergunta: str, contexto: str = "") -> Decisao:
        tokens_pergunta = self.tokenizador.contar(pergunta)
        features = _features(self.embedder, [pergunta], [tokens_pergunta])
        probabilidade = _probabilidade_barato(self.modelo_a, features)
        faixa = _faixa_prevista(self.modelo_b, features)

        limiar = self.cfg.limiar_probabilidade
        if probabilidade >= limiar:
            return Decisao(
                papel=Papel.BARATO,
                motivo=(
                    f"probabilidade de o barato bastar: {probabilidade:.3f} "
                    f"(limiar {limiar:.3f})"
                ),
                faixa_prevista=faixa,
            )

        return Decisao(
            papel=Papel.FORTE,
            motivo=(
                f"probabilidade de o barato bastar: {probabilidade:.3f} "
                f"(limiar {limiar:.3f})"
            ),
            faixa_prevista=faixa,
            limite_tokens=_limite_do_forte(self.modelo_b, faixa),
        )


def montar_features_linhas(
    linhas: Sequence[LinhaDataset],
    embedder: Embedder,
    tokenizador: Tokenizador,
) -> np.ndarray:
    """Monta as features do v2 para as linhas do dataset."""
    lista = list(linhas)
    if not lista:
        raise ValueError("Não é possível montar features sem linhas.")

    embeddings = embedder.embutir([linha.pergunta for linha in lista])
    tokens = [tokenizador.contar(linha.pergunta) for linha in lista]
    return montar_features(embeddings, tokens)


def treinar_modelo_v2(
    linhas: Sequence[LinhaDataset],
    embedder: Embedder,
    tokenizador: Tokenizador,
    cfg: Config,
) -> ModeloRoteador:
    """Treina os classificadores A e B e devolve o bundle do roteador v2."""
    lista = list(linhas)
    if not lista:
        raise ValueError("Não é possível treinar o v2 sem linhas no dataset.")

    cortes = calcular_cortes(
        [linha.tokens_resposta_forte for linha in lista],
        cfg.percentis_faixas,
    )
    features = montar_features_linhas(lista, embedder, tokenizador)
    nome_embedder = str(getattr(embedder, "nome", ""))

    modelo_a = treinar_a(
        features,
        [barato_basta(linha) for linha in lista],
        embedder_nome=nome_embedder,
    )
    modelo_b = treinar_b(
        features,
        [faixa_da_linha(linha, cortes).value for linha in lista],
        embedder_nome=nome_embedder,
        cortes=cortes,
    )
    return ModeloRoteador(
        modelo_a=modelo_a,
        modelo_b=modelo_b,
        cortes=cortes,
        embedder_nome=nome_embedder,
    )


def _features(
    embedder: Embedder,
    textos: Sequence[str],
    tokens: Sequence[int | float],
) -> np.ndarray:
    embeddings = embedder.embutir(textos)
    return montar_features(embeddings, tokens)


def _probabilidade_barato(modelo_a: Any, features: np.ndarray) -> float:
    try:
        probabilidades = np.asarray(modelo_a.predict_proba(features), dtype=float)
    except Exception as exc:
        raise ValueError(
            f"Não foi possível obter a probabilidade do classificador A: {exc}"
        ) from exc

    if probabilidades.ndim != 2 or probabilidades.shape[0] == 0:
        raise ValueError(
            "O classificador A devolveu probabilidades com formato inválido."
        )

    classes = getattr(modelo_a, "classes_", None)
    if classes is None:
        indice = probabilidades.shape[1] - 1
    else:
        if len(classes) != probabilidades.shape[1]:
            raise ValueError(
                "O classificador A devolveu probabilidades incompatíveis "
                "com as classes treinadas."
            )
        indice = _indice_da_classe_verdadeira(classes)

    return float(probabilidades[0, indice])


def _indice_da_classe_verdadeira(classes: Sequence[Any]) -> int:
    for indice, classe in enumerate(classes):
        if isinstance(classe, (bool, np.bool_)) and bool(classe):
            return indice
    for indice, classe in enumerate(classes):
        if classe is True or classe == True:  # noqa: E712
            return indice
    raise ValueError(
        "O classificador A não tem a classe verdadeira de 'o barato basta'."
    )


def _faixa_prevista(modelo_b: Any, features: np.ndarray) -> Faixa:
    try:
        previstos = modelo_b.predict(features)
    except Exception as exc:
        raise ValueError(
            f"Não foi possível obter a faixa do classificador B: {exc}"
        ) from exc

    if isinstance(previstos, (Faixa, str)):
        valor = previstos
    else:
        valores = list(previstos)
        if not valores:
            raise ValueError("O classificador B não devolveu nenhuma faixa.")
        valor = valores[0]

    if isinstance(valor, Faixa):
        return valor
    try:
        return Faixa(valor)
    except ValueError:
        raise ValueError(
            f"O classificador B devolveu uma faixa inválida: {valor!r}."
        ) from None


def _limite_do_forte(modelo_b: Any, faixa: Faixa) -> int | None:
    cortes = getattr(modelo_b, "cortes", None)
    if cortes is None:
        if faixa == Faixa.MEDIA:
            raise ValueError(
                "O classificador B não tem os cortes das faixas. "
                "Carregue um modelo treinado antes de usar o roteador v2."
            )
        return None
    return limite_por_faixa(faixa, cortes)
