import csv
import json

import pytest

from roteador.config import Config, carregar_config
from roteador.dados.carregador import LinhaDataset, Nota
from roteador.ml.limiar import (
    COLUNAS_CALIBRACAO,
    ResultadoLimiar,
    avaliar_limiar,
    calibrar_limiar_tokens,
    candidatos_limiar,
    dividir_calibracao,
    salvar_calibracao,
    salvar_config_limiar,
    selecionar_limiar,
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
