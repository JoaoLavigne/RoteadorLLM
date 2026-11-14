import numpy as np
import pytest

from roteador.ml.classificadores import (
    CLASSES_B,
    ErroModelo,
    ModeloConstante,
    ModeloRoteador,
    avaliar,
    carregar,
    carregar_modelo,
    montar_features,
    salvar,
    salvar_modelo,
    treinar_a,
    treinar_b,
)
from roteador.ml.embedder import EmbedderFalso


def _dados_binarios():
    tokens = np.arange(1, 201, dtype=float)
    embeddings = EmbedderFalso(dimensao=4).embutir(
        [f"pergunta {int(valor)}" for valor in tokens]
    )
    features = montar_features(embeddings, tokens)
    rotulos = tokens < 100
    return features, rotulos


def _dados_faixas():
    tokens = np.arange(1, 301, dtype=float)
    embeddings = EmbedderFalso(dimensao=4).embutir(
        [f"pergunta {int(valor)}" for valor in tokens]
    )
    features = montar_features(embeddings, tokens)
    rotulos = np.where(tokens < 100, "curta", np.where(tokens < 200, "media", "longa"))
    return features, rotulos


def test_montar_features_junta_embedding_e_log1p():
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    features = montar_features(embeddings, [0, 3])

    assert features.shape == (2, 3)
    assert np.allclose(features[:, :2], embeddings)
    assert np.allclose(features[:, 2], [0.0, np.log1p(3)])


def test_montar_features_valida_quantidades():
    with pytest.raises(ValueError, match="quantidade"):
        montar_features(np.zeros((2, 3)), [1])
    with pytest.raises(ValueError, match="bidimensional"):
        montar_features(np.zeros(3), [1])
    with pytest.raises(ValueError, match="negativa"):
        montar_features(np.zeros((1, 3)), [-1])


def test_treinar_a_aprende_regra_simples():
    features, rotulos = _dados_binarios()

    modelo = treinar_a(features, rotulos, embedder_nome="falso:4")
    metricas = avaliar(modelo, features, rotulos)

    assert metricas["acuracia"] > 0.9
    assert modelo.embedder_nome == "falso:4"
    assert (
        np.mean(
            np.asarray(modelo.predict(features), dtype=bool)
            == np.asarray(rotulos, dtype=bool)
        )
        > 0.9
    )


def test_treinar_b_aprende_faixas_e_matriz_tem_tres_classes():
    features, rotulos = _dados_faixas()

    modelo = treinar_b(features, rotulos, embedder_nome="falso:4", cortes=(100, 200))
    metricas = avaliar(modelo, features, rotulos, classes=CLASSES_B)

    assert metricas["acuracia"] > 0.9
    assert metricas["classes"] == list(CLASSES_B)
    assert len(metricas["matriz_confusao"]) == 3
    assert all(len(linha) == 3 for linha in metricas["matriz_confusao"])
    assert set(metricas["precisao"]) == set(CLASSES_B)
    assert set(metricas["revocacao"]) == set(CLASSES_B)
    assert modelo.cortes == (100, 200)


def test_treinar_com_uma_classe_usa_modelo_constante():
    features, _ = _dados_binarios()
    rotulos = np.ones(len(features), dtype=bool)

    modelo = treinar_a(features, rotulos)

    assert isinstance(modelo, ModeloConstante)
    assert np.all(modelo.predict(features))
    assert modelo.predict_proba(features).shape == (len(features), 2)


def test_salvar_e_carregar_mantem_predicao(tmp_path):
    features, rotulos = _dados_binarios()
    modelo = treinar_a(features, rotulos, embedder_nome="falso:4")
    caminho = tmp_path / "modelo_a.joblib"

    salvar(modelo, caminho)
    carregado = carregar(caminho, embedder_esperado="falso:4")

    assert np.array_equal(modelo.predict(features), carregado.predict(features))
    assert carregado.embedder_nome == "falso:4"


def test_carregar_recusa_modelo_de_outro_embedder(tmp_path):
    features, rotulos = _dados_binarios()
    modelo = treinar_a(features, rotulos, embedder_nome="falso:4")
    caminho = tmp_path / "modelo_a.joblib"
    salvar(modelo, caminho)

    with pytest.raises(ErroModelo, match="outro embedder"):
        carregar(caminho, embedder_esperado="falso:8")


def test_carregar_aceita_objeto_embedder(tmp_path):
    features, rotulos = _dados_binarios()
    embedder = EmbedderFalso(dimensao=4)
    modelo = treinar_a(features, rotulos, embedder_nome=embedder.nome)
    caminho = tmp_path / "modelo_a.joblib"
    salvar(modelo, caminho)

    carregado = carregar(caminho, embedder_esperado=embedder)

    assert carregado.embedder_nome == embedder.nome


def test_carregar_modelo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        carregar(tmp_path / "sumiu.joblib")


def test_salvar_e_carregar_bundle(tmp_path):
    features, rotulos = _dados_binarios()
    modelo_a = treinar_a(features, rotulos, embedder_nome="falso:4")
    features_b, rotulos_b = _dados_faixas()
    modelo_b = treinar_b(
        features_b,
        rotulos_b,
        embedder_nome="falso:4",
        cortes=(100, 200),
    )
    bundle = ModeloRoteador(
        modelo_a=modelo_a,
        modelo_b=modelo_b,
        cortes=(100, 200),
        embedder_nome="falso:4",
    )
    caminho = tmp_path / "modelo_v2.joblib"

    salvar_modelo(bundle, caminho)
    carregado = carregar_modelo(caminho, embedder_esperado="falso:4")

    assert isinstance(carregado, ModeloRoteador)
    assert carregado.cortes == (100, 200)
    assert np.array_equal(
        carregado.modelo_a.predict(features), modelo_a.predict(features)
    )


def test_carregar_modelo_recusa_bundle_ausente(tmp_path):
    caminho = tmp_path / "outro.joblib"
    salvar({"modelo_a": None}, caminho)

    with pytest.raises(ErroModelo, match="ModeloRoteador"):
        carregar_modelo(caminho)
