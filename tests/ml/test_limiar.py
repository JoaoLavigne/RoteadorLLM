import csv
import json

import numpy as np
import pytest

from roteador.config import Config, carregar_config
from roteador.dados.carregador import LinhaDataset, Nota
from roteador.ml.classificadores import ModeloRoteador
from roteador.ml.embedder import EmbedderFalso
from roteador.ml.limiar import (
    COLUNAS_CALIBRACAO,
    COLUNAS_CALIBRACAO_PROBABILIDADE,
    CalibracaoProbabilidade,
    ResultadoLimiar,
    ResultadoLimiarProbabilidade,
    avaliar_limiar,
    avaliar_limiar_probabilidade,
    calibrar_limiar_probabilidade,
    calibrar_limiar_tokens,
    candidatos_limiar,
    candidatos_limiar_probabilidade,
    dividir_calibracao,
    salvar_calibracao,
    salvar_calibracao_probabilidade,
    salvar_config_limiar,
    salvar_config_limiar_probabilidade,
    selecionar_limiar,
    selecionar_limiar_probabilidade,
)


class TokenizadorPorTexto:
    def __init__(self, mapa):
        self.mapa = mapa

    def contar(self, texto):
        return self.mapa.get(texto, len(texto))


class TokenizadorFixo:
    def __init__(self, tokens):
        self.tokens = tokens

    def contar(self, texto):
        return self.tokens


class ModeloAProbabilidades:
    """Devolve as probabilidades na ordem em que foram informadas."""

    def __init__(self, probabilidades):
        self.probabilidades = list(probabilidades)
        self.classes_ = np.array([False, True])

    def predict_proba(self, features):
        quantidade = len(features)
        valores = np.asarray(
            self.probabilidades[:quantidade], dtype=float
        )
        self.probabilidades = self.probabilidades[quantidade:]
        return np.column_stack([1 - valores, valores])


def _modelo(probabilidades):
    return ModeloRoteador(
        modelo_a=ModeloAProbabilidades(probabilidades),
        modelo_b=None,
        cortes=(0.0, 0.0),
        embedder_nome="teste",
    )


def _linhas_probabilidade(scores, probabilidades=None):
    linhas = []
    for indice, (score_barato, score_forte) in enumerate(scores):
        linhas.append(
            _linha(
                id=str(indice),
                documento=f"doc-{indice}",
                pergunta=f"pergunta {indice}",
                score_barato=score_barato,
                score_forte=score_forte,
            )
        )
    return linhas


def _resultado_prob(limiar, qualidade_relativa_pct, custo=100.0):
    return ResultadoLimiarProbabilidade(
        limiar=limiar,
        qualidade_media=qualidade_relativa_pct / 100,
        custo_estimado_usd=custo,
        qualidade_relativa_pct=qualidade_relativa_pct,
        custo_relativo_pct=100.0,
        economia_pct=0.0,
        linhas=1,
    )


def _cfg(**extra):
    valores = {
        "preco_ref": {
            "barato": (1_000_000.0, 2_000_000.0),
            "forte": (4_000_000.0, 8_000_000.0),
        },
        "janela_contexto_barato": 1000,
        "limiar_tokens_pergunta": 100,
    }
    valores.update(extra)
    return Config(**valores)


def _linha(**valores):
    padrao = {
        "id": "1",
        "documento": "doc-a",
        "tipo": "aberta",
        "pergunta": "pergunta",
        "nota_barato": Nota.CORRETA,
        "nota_forte": Nota.CORRETA,
        "tokens_resposta_barato": 10,
        "tokens_resposta_forte": 100,
    }
    padrao.update(valores)
    return LinhaDataset(**padrao)


def _dataset_documentos(quantidade_documentos=10, por_documento=2, **extra):
    linhas = []
    for documento in range(quantidade_documentos):
        for indice in range(por_documento):
            linhas.append(
                _linha(
                    id=f"{documento}-{indice}",
                    documento=f"doc-{documento}",
                    pergunta=f"pergunta {documento} {indice}",
                    **extra,
                )
            )
    return linhas


def _resultado(limiar, qualidade_relativa_pct):
    return ResultadoLimiar(
        limiar=limiar,
        qualidade_media=qualidade_relativa_pct / 200,
        custo_estimado_usd=0.0,
        qualidade_relativa_pct=qualidade_relativa_pct,
        custo_relativo_pct=100.0,
        economia_pct=0.0,
        linhas=1,
    )


def test_candidatos_incluem_percentis_e_valores_fixos():
    linhas = [
        _linha(id=str(n), documento=f"doc-{n}", pergunta="x" * n)
        for n in range(1, 21)
    ]

    candidatos = candidatos_limiar(linhas, TokenizadorPorTexto({}))

    assert candidatos == sorted(set(candidatos))
    assert 11 in candidatos
    assert 20 in candidatos
    for fixo in (100, 200, 300, 500):
        assert fixo in candidatos


def test_avaliar_limiar_simula_a_regra_do_v1():
    linhas = [
        _linha(
            id="1",
            pergunta="curta",
            nota_barato=Nota.CORRETA,
            nota_forte=Nota.CORRETA,
        ),
        _linha(
            id="2",
            pergunta="longa",
            nota_barato=Nota.ERRADA,
            nota_forte=Nota.CORRETA,
        ),
    ]

    resultado = avaliar_limiar(
        linhas, _cfg(), TokenizadorPorTexto({"curta": 5, "longa": 20}), 10
    )

    assert resultado.limiar == 10
    assert resultado.qualidade_media == pytest.approx(1.0)
    assert resultado.custo_estimado_usd == pytest.approx(905.0)
    assert resultado.custo_relativo_pct == pytest.approx(100 * 905 / 1700)


def test_limiar_maior_manda_mais_perguntas_ao_barato():
    linhas = [
        _linha(id="1", pergunta="curta"),
        _linha(id="2", pergunta="longa"),
    ]
    tokenizador = TokenizadorPorTexto({"curta": 5, "longa": 20})

    apertado = avaliar_limiar(linhas, _cfg(), tokenizador, 10)
    folgado = avaliar_limiar(linhas, _cfg(), tokenizador, 100)

    assert folgado.custo_estimado_usd < apertado.custo_estimado_usd


def test_seleciona_o_maior_limiar_que_mantem_a_meta():
    resultados = [
        _resultado(100, 94.0),
        _resultado(200, 96.0),
        _resultado(300, 95.0),
        _resultado(500, 90.0),
    ]

    escolhido, aviso = selecionar_limiar(resultados, 95.0)

    assert escolhido.limiar == 300
    assert aviso is None


def test_sem_candidato_viavel_escolhe_o_melhor_e_avisa():
    resultados = [_resultado(100, 80.0), _resultado(200, 90.0)]

    escolhido, aviso = selecionar_limiar(resultados, 95.0)

    assert escolhido.limiar == 200
    assert aviso is not None
    assert "Nenhum limiar" in aviso


def test_dividir_calibracao_nao_vaza_documentos():
    linhas = _dataset_documentos()

    treino, validacao, teste = dividir_calibracao(linhas)

    documentos_treino = {linha.documento for linha in treino}
    documentos_validacao = {linha.documento for linha in validacao}
    documentos_teste = {linha.documento for linha in teste}

    assert not documentos_treino & documentos_validacao
    assert not documentos_treino & documentos_teste
    assert not documentos_validacao & documentos_teste
    assert documentos_treino | documentos_validacao | documentos_teste == {
        linha.documento for linha in linhas
    }
    assert len(documentos_teste) == 2
    assert len(documentos_validacao) == 2


def test_dividir_calibracao_exige_tres_documentos():
    linhas = _dataset_documentos(quantidade_documentos=2)

    with pytest.raises(ValueError, match="pelo menos 3 documentos"):
        dividir_calibracao(linhas)


def test_calibrar_aprova_quando_o_barato_iguala_o_forte():
    linhas = _dataset_documentos()

    primeira = calibrar_limiar_tokens(
        linhas, _cfg(), TokenizadorFixo(1)
    )
    segunda = calibrar_limiar_tokens(
        linhas, _cfg(), TokenizadorFixo(1)
    )

    assert primeira == segunda
    assert primeira.aprovado
    assert primeira.limiar_escolhido == 500
    assert primeira.teste is not None
    assert primeira.teste.qualidade_relativa_pct == 100.0
    assert primeira.aviso_selecao is None
    assert primeira.aviso_teste is None
    assert primeira.linhas_treino > 0
    assert primeira.linhas_validacao > 0
    assert primeira.linhas_teste > 0


def test_calibrar_reprova_quando_o_barato_perde_do_forte():
    linhas = _dataset_documentos(
        nota_barato=Nota.ERRADA,
        nota_forte=Nota.CORRETA,
    )

    calibracao = calibrar_limiar_tokens(
        linhas, _cfg(), TokenizadorFixo(1)
    )

    assert not calibracao.aprovado
    assert calibracao.aviso_selecao is not None
    assert calibracao.aviso_teste is not None


def test_salvar_calibracao_grava_uma_linha_por_candidato(tmp_path):
    linhas = _dataset_documentos()
    calibracao = calibrar_limiar_tokens(
        linhas, _cfg(), TokenizadorFixo(1)
    )
    caminho = tmp_path / "calibracao.csv"

    salvar_calibracao(calibracao, caminho)

    with caminho.open(encoding="utf-8", newline="") as aberto:
        registros = list(csv.DictReader(aberto))

    assert list(registros[0]) == list(COLUNAS_CALIBRACAO)
    assert len(registros) == len(calibracao.resultados)
    escolhidos = [
        registro
        for registro in registros
        if registro["escolhido"] == "1"
    ]
    assert len(escolhidos) == 1
    assert int(escolhidos[0]["limiar_tokens_pergunta"]) == (
        calibracao.limiar_escolhido
    )


def test_salvar_config_limiar_pode_ser_carregado(tmp_path):
    linhas = _dataset_documentos()
    cfg = _cfg(tokenizador="tiktoken:o200k_base")
    calibracao = calibrar_limiar_tokens(
        linhas, cfg, TokenizadorFixo(1)
    )
    caminho = tmp_path / "config_limiar_tokens.json"

    salvar_config_limiar(calibracao, cfg, caminho)
    carregada = carregar_config(caminho)

    assert carregada.limiar_tokens_pergunta == calibracao.limiar_escolhido
    assert carregada.tokenizador == "tiktoken:o200k_base"
    assert set(json.loads(caminho.read_text(encoding="utf-8"))) == {
        "limiar_tokens_pergunta",
        "tokenizador",
    }


def test_calibrar_escolhe_limiar_com_economia_no_cenario_simulado():
    linhas = []
    for documento in range(10):
        linhas.append(
            _linha(
                id=f"{documento}-curta",
                documento=f"doc-{documento}",
                pergunta="curta",
                nota_barato=Nota.CORRETA,
                nota_forte=Nota.CORRETA,
            )
        )
        linhas.append(
            _linha(
                id=f"{documento}-longa",
                documento=f"doc-{documento}",
                pergunta="longa",
                nota_barato=Nota.ERRADA,
                nota_forte=Nota.CORRETA,
            )
        )

    calibracao = calibrar_limiar_tokens(
        linhas,
        Config(),
        TokenizadorPorTexto({"curta": 10, "longa": 100}),
    )

    assert calibracao.aprovado
    assert calibracao.limiar_escolhido == 82
    assert calibracao.teste is not None
    assert calibracao.teste.qualidade_relativa_pct >= 95.0
    assert calibracao.teste.economia_pct == pytest.approx(45.7, abs=0.1)


def test_candidatos_limiar_probabilidade_cobrem_zero_e_um():
    candidatos = candidatos_limiar_probabilidade(0.25)

    assert candidatos == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_candidatos_limiar_probabilidade_com_passo_invalido_falha():
    for passo in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match="passo"):
            candidatos_limiar_probabilidade(passo)


def test_avaliar_limiar_probabilidade_simula_o_v2():
    linhas = _linhas_probabilidade(
        [(0.8, 0.9), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
    )

    resultado = avaliar_limiar_probabilidade(
        linhas,
        [0.9, 0.8, 0.2, 0.1],
        _cfg(),
        TokenizadorFixo(10),
        0.5,
    )

    assert resultado.limiar == 0.5
    assert resultado.qualidade_media == pytest.approx(0.825)
    assert resultado.qualidade_relativa_pct == pytest.approx(97.058823, rel=1e-5)
    assert resultado.custo_estimado_usd == pytest.approx(1740.0)
    assert resultado.custo_relativo_pct == pytest.approx(51.785714, rel=1e-5)
    assert resultado.economia_pct == pytest.approx(48.214286, rel=1e-5)


def test_avaliar_limiar_probabilidade_valida_entradas():
    linhas = _linhas_probabilidade([(1.0, 1.0)])

    with pytest.raises(ValueError, match="difere"):
        avaliar_limiar_probabilidade(
            linhas, [0.5, 0.6], _cfg(), TokenizadorFixo(1), 0.5
        )
    with pytest.raises(ValueError, match="entre 0 e 1"):
        avaliar_limiar_probabilidade(
            linhas, [0.5], _cfg(), TokenizadorFixo(1), 1.5
        )


def test_seleciona_o_menor_limiar_que_mantem_a_meta():
    resultados = [
        _resultado_prob(0.1, 94.0),
        _resultado_prob(0.2, 96.0, custo=90.0),
        _resultado_prob(0.3, 96.0, custo=80.0),
        _resultado_prob(0.5, 90.0),
    ]

    escolhido, aviso = selecionar_limiar_probabilidade(resultados, 95.0)

    assert escolhido.limiar == 0.2
    assert aviso is None


def test_empate_de_metricas_prefere_o_limiar_mais_alto():
    resultados = [
        _resultado_prob(0.2, 96.0, custo=90.0),
        _resultado_prob(0.4, 96.0, custo=90.0),
        _resultado_prob(0.8, 90.0),
    ]

    escolhido, aviso = selecionar_limiar_probabilidade(resultados, 95.0)

    assert escolhido.limiar == 0.4
    assert aviso is None


def test_sem_limiar_viavel_escolhe_o_melhor_e_avisa():
    resultados = [
        _resultado_prob(0.1, 80.0),
        _resultado_prob(0.2, 90.0),
    ]

    escolhido, aviso = selecionar_limiar_probabilidade(resultados, 95.0)

    assert escolhido.limiar == 0.2
    assert aviso is not None
    assert "Nenhum limiar" in aviso


def test_calibrar_probabilidade_aprova_quando_o_teste_mantem_a_meta():
    validacao = _linhas_probabilidade(
        [(0.8, 0.9), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
    )
    teste = _linhas_probabilidade(
        [(0.9, 1.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0)]
    )
    modelo = _modelo([0.9, 0.8, 0.2, 0.1, 0.9, 0.1, 0.9, 0.1])

    calibracao = calibrar_limiar_probabilidade(
        validacao,
        teste,
        modelo,
        EmbedderFalso(dimensao=4),
        TokenizadorFixo(10),
        _cfg(),
        limiares=[0.5],
    )

    assert calibracao.aprovado
    assert calibracao.limiar_escolhido == 0.5
    assert calibracao.aviso_selecao is None
    assert calibracao.aviso_teste is None
    assert calibracao.teste is not None
    assert calibracao.teste.qualidade_relativa_pct == pytest.approx(97.5)
    assert calibracao.linhas_validacao == 4
    assert calibracao.linhas_teste == 4


def test_calibrar_probabilidade_reprova_quando_o_teste_cai():
    validacao = _linhas_probabilidade(
        [(1.0, 1.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0)]
    )
    teste = _linhas_probabilidade(
        [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    )
    modelo = _modelo([0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9])

    calibracao = calibrar_limiar_probabilidade(
        validacao,
        teste,
        modelo,
        EmbedderFalso(dimensao=4),
        TokenizadorFixo(10),
        _cfg(),
        limiares=[0.5],
    )

    assert not calibracao.aprovado
    assert calibracao.aviso_teste is not None
    assert calibracao.teste is not None
    assert calibracao.teste.qualidade_relativa_pct == 0.0


def test_salvar_calibracao_probabilidade_grava_uma_linha_por_candidato(tmp_path):
    calibracao = CalibracaoProbabilidade(
        limiar_escolhido=0.2,
        qualidade_minima_pct=95.0,
        resultados=(
            _resultado_prob(0.1, 94.0),
            _resultado_prob(0.2, 96.0),
        ),
        linhas_validacao=4,
        linhas_teste=4,
    )
    caminho = tmp_path / "calibracao_probabilidade.csv"

    salvar_calibracao_probabilidade(calibracao, caminho)

    with caminho.open(encoding="utf-8", newline="") as aberto:
        registros = list(csv.DictReader(aberto))

    assert list(registros[0]) == list(COLUNAS_CALIBRACAO_PROBABILIDADE)
    assert len(registros) == 2
    assert [registro["escolhido"] for registro in registros] == ["0", "1"]


def test_salvar_config_limiar_probabilidade_pode_ser_carregado(tmp_path):
    cfg = _cfg(tokenizador="tiktoken:o200k_base")
    calibracao = CalibracaoProbabilidade(
        limiar_escolhido=0.42,
        qualidade_minima_pct=95.0,
        resultados=(_resultado_prob(0.42, 96.0),),
        linhas_validacao=4,
        linhas_teste=4,
    )
    caminho = tmp_path / "config_limiar_probabilidade.json"

    salvar_config_limiar_probabilidade(calibracao, cfg, caminho)
    carregada = carregar_config(caminho)

    assert carregada.limiar_probabilidade == pytest.approx(0.42)
    assert carregada.tokenizador == "tiktoken:o200k_base"
    assert set(json.loads(caminho.read_text(encoding="utf-8"))) == {
        "limiar_probabilidade",
        "tokenizador",
    }
