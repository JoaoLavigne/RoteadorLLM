import numpy as np
import pytest

from roteador.ml.embedder import (
    Embedder,
    EmbedderFalso,
    EmbedderOllama,
    ErroEmbedder,
)


class RespostaFake:
    def __init__(self, dados=None, status_code=200, texto=""):
        self._dados = dados
        self.status_code = status_code
        self.text = texto

    def json(self):
        if isinstance(self._dados, Exception):
            raise self._dados
        return self._dados


def test_embedder_falso_e_deterministico_e_respeita_dimensao():
    embedder = EmbedderFalso(dimensao=5)

    vetores = embedder.embutir(["alpha", "beta", "alpha"])
    de_novo = EmbedderFalso(dimensao=5).embutir(["alpha", "beta", "alpha"])

    assert vetores.shape == (3, 5)
    assert np.array_equal(vetores, de_novo)
    assert np.array_equal(vetores[0], vetores[2])
    assert not np.array_equal(vetores[0], vetores[1])
    assert isinstance(embedder, Embedder)


def test_embedder_falso_texto_vazio():
    vetores = EmbedderFalso(dimensao=3).embutir([])

    assert vetores.shape == (0, 3)


def test_embedder_falso_rejeita_dimensao_invalida():
    with pytest.raises(ValueError, match="dimensão"):
        EmbedderFalso(dimensao=0)


def test_embedder_ollama_monta_payload_e_devolve_matriz():
    chamadas = []

    def post(url, json):
        chamadas.append((url, json))
        return RespostaFake({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    embedder = EmbedderOllama(modelo="modelo-x", host="http://host:1234/", post=post)
    vetores = embedder.embutir(["a", "b"])

    assert vetores.shape == (2, 2)
    assert np.allclose(vetores, [[0.1, 0.2], [0.3, 0.4]])
    assert chamadas == [
        (
            "http://host:1234/api/embed",
            {"model": "modelo-x", "input": ["a", "b"]},
        )
    ]
    assert embedder.nome == "modelo-x"


def test_embedder_ollama_usa_host_do_ambiente(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://outro:9999")
    embedder = EmbedderOllama(modelo="m", post=lambda *a, **k: None)

    assert embedder.host == "http://outro:9999"


def test_embedder_ollama_texto_vazio_nao_chama_post():
    def post(*args, **kwargs):
        raise AssertionError("não deveria chamar o Ollama")

    vetores = EmbedderOllama(post=post).embutir([])

    assert vetores.shape == (0, 0)


def test_embedder_ollama_erro_http():
    embedder = EmbedderOllama(
        post=lambda *a, **k: RespostaFake({}, status_code=500, texto="falhou")
    )

    with pytest.raises(ErroEmbedder, match="HTTP 500"):
        embedder.embutir(["a"])


def test_embedder_ollama_erro_de_rede():
    def post(*args, **kwargs):
        raise RuntimeError("sem conexão")

    with pytest.raises(ErroEmbedder, match="sem conexão"):
        EmbedderOllama(post=post).embutir(["a"])


def test_embedder_ollama_json_invalido():
    embedder = EmbedderOllama(
        post=lambda *a, **k: RespostaFake(ValueError("json quebrado"))
    )

    with pytest.raises(ErroEmbedder, match="JSON"):
        embedder.embutir(["a"])


def test_embedder_ollama_resposta_sem_um_vetor_por_texto():
    embedder = EmbedderOllama(
        post=lambda *a, **k: RespostaFake({"embeddings": [[0.1, 0.2]]})
    )

    with pytest.raises(ErroEmbedder, match="um vetor por texto"):
        embedder.embutir(["a", "b"])


def test_embedder_ollama_vetores_irregulares():
    embedder = EmbedderOllama(
        post=lambda *a, **k: RespostaFake({"embeddings": [[0.1], [0.2, 0.3]]})
    )

    with pytest.raises(ErroEmbedder, match="matriz"):
        embedder.embutir(["a", "b"])
