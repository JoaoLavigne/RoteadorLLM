import numpy as np
import pytest

from roteador.config import Config
from roteador.dados.carregador import LinhaDataset, Nota
from roteador.ml.embedder import EmbedderFalso
from roteador.ml.roteador_v2 import (
    RoteadorV2,
    montar_features_linhas,
    treinar_modelo_v2,
)
from roteador.tipos import Faixa, Papel, Roteador


class TokenizadorFixo:
    def __init__(self, tokens=10):
        self.tokens = tokens

    def contar(self, texto):
        return self.tokens


class ModeloA:
    def __init__(self, probabilidade):
        self.probabilidade = probabilidade
        self.classes_ = np.array([False, True])

    def predict_proba(self, features):
        probabilidades = np.full(len(features), self.probabilidade, dtype=float)
        return np.column_stack([1 - probabilidades, probabilidades])


class ModeloB:
    def __init__(self, faixa, cortes=(100.0, 200.0)):
        self.faixa = faixa
        self.cortes = cortes

    def predict(self, features):
        return np.array([self.faixa for _ in features], dtype=object)


def _cfg(**extra):
    valores = {"limiar_probabilidade": 0.5}
    valores.update(extra)
    return Config(**valores)


def _roteador(probabilidade, faixa=Faixa.CURTA, cortes=(100.0, 200.0), **extra):
    return RoteadorV2(
        EmbedderFalso(dimensao=4),
        ModeloA(probabilidade),
        ModeloB(faixa, cortes),
        TokenizadorFixo(10),
        _cfg(**extra),
    )


@pytest.mark.parametrize(
    ("probabilidade", "papel"),
    [
        (0.49, Papel.FORTE),
        (0.5, Papel.BARATO),
        (0.51, Papel.BARATO),
    ],
)
def test_limiar_nas_bordas(probabilidade, papel):
    decisao = _roteador(probabilidade).decidir("pergunta")

    assert decisao.papel == papel
    assert f"{probabilidade:.3f}" in decisao.motivo


def test_forte_com_faixa_media_recebe_o_limite():
    decisao = _roteador(
        0.1, faixa=Faixa.MEDIA, cortes=(100.0, 250.0)
    ).decidir("pergunta")

    assert decisao.papel == Papel.FORTE
    assert decisao.faixa_prevista == Faixa.MEDIA
    assert decisao.limite_tokens == 250


@pytest.mark.parametrize("faixa", [Faixa.CURTA, Faixa.LONGA])
def test_forte_com_faixa_curta_ou_longa_nao_recebe_limite(faixa):
    decisao = _roteador(0.1, faixa=faixa).decidir("pergunta")

    assert decisao.papel == Papel.FORTE
    assert decisao.faixa_prevista == faixa
    assert decisao.limite_tokens is None


def test_barato_nunca_recebe_limite():
    decisao = _roteador(
        0.9, faixa=Faixa.MEDIA, cortes=(100.0, 250.0)
    ).decidir("pergunta")

    assert decisao.papel == Papel.BARATO
    assert decisao.faixa_prevista == Faixa.MEDIA
    assert decisao.limite_tokens is None


def test_forte_medio_sem_cortes_falha_com_mensagem_clara():
    roteador = RoteadorV2(
        EmbedderFalso(dimensao=4),
        ModeloA(0.1),
        ModeloB(Faixa.MEDIA, cortes=None),
        TokenizadorFixo(10),
        _cfg(),
    )

    with pytest.raises(ValueError, match="cortes"):
        roteador.decidir("pergunta")


def test_limiar_invalido_falha_na_construcao():
    with pytest.raises(ValueError, match="entre 0 e 1"):
        _roteador(0.5, limiar_probabilidade=1.5)


def test_satisfaz_o_protocol_roteador():
    assert isinstance(_roteador(0.5), Roteador)


def test_treinar_e_decidir_com_embedder_falso():
    linhas = [
        LinhaDataset(
            id=str(indice),
            documento=f"doc-{indice}",
            tipo="aberta",
            pergunta=pergunta,
            nota_barato=nota_barato,
            nota_forte=nota_forte,
            tokens_resposta_barato=10,
            tokens_resposta_forte=tokens_forte,
        )
        for indice, (pergunta, nota_barato, nota_forte, tokens_forte) in enumerate(
            [
                ("curta simples", Nota.CORRETA, Nota.CORRETA, 50),
                ("curta igual", Nota.PARCIAL, Nota.PARCIAL, 60),
                ("media dificil", Nota.ERRADA, Nota.CORRETA, 150),
                ("longa complexa", Nota.ERRADA, Nota.CORRETA, 300),
            ]
        )
    ]
    embedder = EmbedderFalso(dimensao=4)
    tokenizador = TokenizadorFixo(50)
    modelo = treinar_modelo_v2(linhas, embedder, tokenizador, _cfg())

    assert modelo.cortes == pytest.approx((60.0, 150.0))
    assert modelo.embedder_nome == embedder.nome
    assert montar_features_linhas(linhas, embedder, tokenizador).shape[0] == 4

    roteador = RoteadorV2(
        embedder,
        modelo.modelo_a,
        modelo.modelo_b,
        tokenizador,
        _cfg(),
    )
    decisao = roteador.decidir("curta simples")

    assert decisao.papel in {Papel.BARATO, Papel.FORTE}
    assert decisao.faixa_prevista in set(Faixa)
