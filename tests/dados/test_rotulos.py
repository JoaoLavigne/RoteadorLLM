import pytest

from roteador.dados.carregador import LinhaDataset, Nota
from roteador.dados.rotulos import (
    barato_basta,
    calcular_cortes,
    dividir_por_documento,
    faixa_da_linha,
    faixa_de,
)
from roteador.tipos import Faixa


def _linha(documento="doc-a", linha_id="1", **valores):
    padrao = {
        "id": linha_id,
        "documento": documento,
        "tipo": "aberta",
        "pergunta": "pergunta",
        "nota_barato": Nota.CORRETA,
        "nota_forte": Nota.CORRETA,
        "tokens_resposta_barato": 10,
        "tokens_resposta_forte": 20,
        "truncada_forte": False,
    }
    padrao.update(valores)
    return LinhaDataset(**padrao)


@pytest.mark.parametrize(
    ("nota_barato", "nota_forte", "esperado"),
    [
        (Nota.CORRETA, Nota.CORRETA, True),
        (Nota.CORRETA, Nota.PARCIAL, True),
        (Nota.CORRETA, Nota.ERRADA, True),
        (Nota.PARCIAL, Nota.CORRETA, False),
        (Nota.PARCIAL, Nota.PARCIAL, True),
        (Nota.PARCIAL, Nota.ERRADA, True),
        (Nota.ERRADA, Nota.CORRETA, False),
        (Nota.ERRADA, Nota.PARCIAL, False),
        (Nota.ERRADA, Nota.ERRADA, True),
    ],
)
def test_barato_basta_com_todas_as_combinacoes(nota_barato, nota_forte, esperado):
    linha = _linha(nota_barato=nota_barato, nota_forte=nota_forte)

    assert barato_basta(linha) is esperado


def test_barato_basta_aceita_dicionario_com_notas_em_texto():
    linha = {"nota_barato": "parcial", "nota_forte": "errada"}

    assert barato_basta(linha) is True


def test_calcular_cortes_com_dois_valores():
    assert calcular_cortes([10, 20], (0.25, 0.75)) == pytest.approx((12.5, 17.5))


def test_calcular_cortes_com_um_valor_nao_quebra():
    assert calcular_cortes([7], (1 / 3, 2 / 3)) == (7.0, 7.0)


def test_calcular_cortes_com_poucos_dados():
    assert calcular_cortes([1, 2, 3, 4], (1 / 3, 2 / 3)) == pytest.approx((2.0, 3.0))


def test_calcular_cortes_sem_dados_falha():
    with pytest.raises(ValueError, match="sem respostas"):
        calcular_cortes([])


@pytest.mark.parametrize(
    "percentis",
    [(0.5,), (0.6, 0.4), (-0.1, 0.5), (0.5, 1.1)],
)
def test_calcular_cortes_com_percentis_invalidos_falha(percentis):
    with pytest.raises(ValueError, match="percentis"):
        calcular_cortes([1, 2, 3], percentis)


@pytest.mark.parametrize(
    ("tokens", "esperado"),
    [(0, Faixa.CURTA), (10, Faixa.CURTA), (11, Faixa.MEDIA), (20, Faixa.MEDIA),
     (21, Faixa.LONGA)],
)
def test_faixa_de_respeita_as_fronteiras(tokens, esperado):
    assert faixa_de(tokens, (10.0, 20.0)) == esperado


@pytest.mark.parametrize(
    ("tokens", "esperado"),
    [(5, Faixa.MEDIA), (15, Faixa.LONGA), (25, Faixa.LONGA)],
)
def test_resposta_truncada_sobe_uma_faixa(tokens, esperado):
    assert faixa_de(tokens, (10.0, 20.0), truncada_forte=True) == esperado


def test_faixa_da_linha_usa_a_resposta_do_forte_e_o_truncamento():
    assert faixa_da_linha(_linha(tokens_resposta_forte=5), (10.0, 20.0)) == Faixa.CURTA
    assert (
        faixa_da_linha(
            _linha(tokens_resposta_forte=5, truncada_forte=True), (10.0, 20.0)
        )
        == Faixa.MEDIA
    )


def test_faixa_de_com_cortes_invertidos_falha():
    with pytest.raises(ValueError, match="crescente"):
        faixa_de(10, (20.0, 10.0))


def _linhas_por_documento(documentos, linhas_por_documento=2):
    linhas = []
    contador = 0
    for documento in documentos:
        for _ in range(linhas_por_documento):
            contador += 1
            linhas.append(_linha(documento=documento, linha_id=str(contador)))
    return linhas


def test_divisao_por_documento_nao_vaza_entre_treino_e_teste():
    linhas = _linhas_por_documento(["a", "b", "c", "d", "e", "f"])

    treino, teste = dividir_por_documento(linhas, fracao_teste=1 / 3, semente=7)

    documentos_treino = {linha.documento for linha in treino}
    documentos_teste = {linha.documento for linha in teste}
    assert documentos_treino.isdisjoint(documentos_teste)
    assert documentos_treino | documentos_teste == {"a", "b", "c", "d", "e", "f"}
    assert len(documentos_teste) == 2
    assert len(treino) + len(teste) == len(linhas)


def test_divisao_por_documento_e_reprodutivel_com_a_mesma_semente():
    linhas = _linhas_por_documento(["a", "b", "c", "d", "e"])

    primeira = dividir_por_documento(linhas, fracao_teste=0.4, semente=123)
    segunda = dividir_por_documento(linhas, fracao_teste=0.4, semente=123)

    assert primeira == segunda


def test_divisao_com_dois_documentos_coloca_um_de_cada_lado():
    linhas = _linhas_por_documento(["a", "b"])

    treino, teste = dividir_por_documento(linhas, fracao_teste=0.5, semente=1)

    assert {linha.documento for linha in treino} != {linha.documento for linha in teste}
    assert len(treino) == 2
    assert len(teste) == 2


def test_divisao_com_um_documento_falha_com_mensagem_clara():
    linhas = _linhas_por_documento(["a"])

    with pytest.raises(ValueError, match="pelo menos 2 documentos"):
        dividir_por_documento(linhas)


@pytest.mark.parametrize("fracao", [0, 1, -0.1, 1.5])
def test_divisao_com_fracao_invalida_falha(fracao):
    with pytest.raises(ValueError, match="entre 0 e 1"):
        dividir_por_documento(_linhas_por_documento(["a", "b"]), fracao_teste=fracao)
