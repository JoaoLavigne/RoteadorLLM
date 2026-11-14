"""Features, treino, avaliação e persistência dos classificadores A e B."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

CLASSES_B = ("curta", "media", "longa")


class ErroModelo(Exception):
    """Falha ao salvar, carregar ou validar um modelo treinado."""


@dataclass
class ModeloRoteador:
    """Bundle com os dois classificadores e os metadados do treino."""

    modelo_a: Any
    modelo_b: Any
    cortes: tuple[float, float]
    embedder_nome: str


class ModeloConstante:
    """Modelo simples para quando o treino tem uma única classe."""

    def __init__(self, classe: Any, classes: Sequence[Any]) -> None:
        self.classe = classe
        self.classes_ = np.asarray(classes)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.classe)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilidades = np.zeros((len(features), len(self.classes_)))
        indice = list(self.classes_).index(self.classe)
        probabilidades[:, indice] = 1.0
        return probabilidades


def montar_features(
    embeddings: np.ndarray, tokens_pergunta: Sequence[int | float]
) -> np.ndarray:
    """Junta o embedding com o log1p da contagem de tokens."""
    matriz = np.asarray(embeddings, dtype=float)
    if matriz.ndim != 2:
        raise ValueError("Os embeddings devem formar uma matriz bidimensional.")

    tokens = np.asarray(tokens_pergunta, dtype=float).reshape(-1)
    if len(tokens) != len(matriz):
        raise ValueError(
            "A quantidade de contagens de tokens deve ser igual à de embeddings."
        )
    if np.any(tokens < 0):
        raise ValueError("A contagem de tokens não pode ser negativa.")

    return np.hstack([matriz, np.log1p(tokens).reshape(-1, 1)])


def treinar_a(
    features: np.ndarray,
    rotulos: Sequence[Any],
    *,
    embedder_nome: str = "",
) -> Any:
    """Treina a regressão logística binária "o barato basta?"."""
    alvos = np.asarray(rotulos)
    modelo = _treinar_logistico(features, alvos, classes=(False, True))
    modelo.embedder_nome = embedder_nome
    return modelo


def treinar_b(
    features: np.ndarray,
    rotulos: Sequence[Any],
    *,
    embedder_nome: str = "",
    cortes: tuple[float, float] | None = None,
    classes: Sequence[Any] = CLASSES_B,
) -> Any:
    """Treina o classificador multiclasse da faixa de tamanho."""
    alvos = np.asarray(rotulos)
    modelo = _treinar_logistico(features, alvos, classes=tuple(classes))
    modelo.embedder_nome = embedder_nome
    modelo.cortes = cortes
    modelo.classes_b = tuple(classes)
    return modelo


def avaliar(
    modelo: Any,
    features: np.ndarray,
    rotulos: Sequence[Any],
    classes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Calcula acurácia, precisão, revocação e matriz de confusão."""
    alvos = np.asarray(rotulos)
    previstos = np.asarray(modelo.predict(features))
    rotulos_classes = _classes_para_avaliacao(modelo, classes, alvos)

    precisao, revocacao, _, _ = precision_recall_fscore_support(
        alvos,
        previstos,
        labels=rotulos_classes,
        average=None,
        zero_division=0,
    )
    matriz = confusion_matrix(alvos, previstos, labels=rotulos_classes)

    return {
        "acuracia": float(accuracy_score(alvos, previstos)),
        "precisao": {
            classe: float(valor)
            for classe, valor in zip(rotulos_classes, precisao)
        },
        "revocacao": {
            classe: float(valor)
            for classe, valor in zip(rotulos_classes, revocacao)
        },
        "matriz_confusao": matriz.tolist(),
        "classes": list(rotulos_classes),
    }


def salvar(objeto: Any, caminho: str | Path) -> None:
    """Grava um modelo ou bundle em disco com joblib."""
    arquivo = Path(caminho)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(objeto, arquivo)


def carregar(caminho: str | Path, embedder_esperado: Any | None = None) -> Any:
    """Carrega um modelo e valida o embedder usado no treino."""
    arquivo = Path(caminho)
    if not arquivo.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {arquivo}")

    try:
        objeto = joblib.load(arquivo)
    except Exception as exc:
        raise ErroModelo(f"Não foi possível carregar o modelo {arquivo}: {exc}") from exc

    if embedder_esperado is not None:
        _validar_embedder(objeto, _nome_do_embedder(embedder_esperado))
    return objeto


def salvar_modelo(modelo: ModeloRoteador, caminho: str | Path) -> None:
    """Grava o bundle dos classificadores A e B."""
    salvar(modelo, caminho)


def carregar_modelo(
    caminho: str | Path, embedder_esperado: Any | None = None
) -> ModeloRoteador:
    """Carrega o bundle dos classificadores A e B."""
    objeto = carregar(caminho, embedder_esperado)
    if not isinstance(objeto, ModeloRoteador):
        raise ErroModelo(
            f"O arquivo {caminho} não contém um ModeloRoteador."
        )
    return objeto


def _treinar_logistico(
    features: np.ndarray,
    rotulos: np.ndarray,
    *,
    classes: Sequence[Any] | None = None,
) -> Any:
    observadas = np.unique(rotulos)
    todas = list(classes) if classes is not None else list(observadas)
    if len(observadas) < 2:
        return ModeloConstante(observadas[0], todas)

    modelo = LogisticRegression(max_iter=1000, solver="lbfgs")
    modelo.fit(np.asarray(features, dtype=float), rotulos)
    return modelo


def _classes_para_avaliacao(
    modelo: Any,
    classes: Sequence[Any] | None,
    rotulos: np.ndarray,
) -> list[Any]:
    if classes is not None:
        return list(classes)
    classes_b = getattr(modelo, "classes_b", None)
    if classes_b is not None:
        return list(classes_b)
    return list(getattr(modelo, "classes_", np.unique(rotulos)))


def _nome_do_embedder(embedder: Any) -> str:
    if isinstance(embedder, str):
        return embedder
    nome = getattr(embedder, "nome", None)
    if nome is None:
        raise ErroModelo("O embedder esperado não tem um nome.")
    return str(nome)


def _validar_embedder(objeto: Any, esperado: str) -> None:
    usado = getattr(objeto, "embedder_nome", None)
    if usado is None and isinstance(objeto, dict):
        usado = objeto.get("embedder_nome")
    if usado is None:
        raise ErroModelo(
            "O modelo não guarda o nome do embedder usado no treino."
        )
    if usado != esperado:
        raise ErroModelo(
            "O modelo foi treinado com outro embedder: "
            f"esperado {esperado!r}, encontrado {usado!r}."
        )
