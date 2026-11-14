import pytest

from roteador.avaliacao.politicas import (
    ResultadoPolitica,
    avaliar_politicas,
    escolha_barato,
)
from roteador.config import Config
from roteador.dados.carregador import LinhaDataset, Nota


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
        "divisor_aproximado": 4,
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


def _dataset():
    return [
        _linha(id="1", nota_barato=Nota.CORRETA, nota_forte=Nota.CORRETA,
               tokens_resposta_barato=10, tokens_resposta_forte=100),
        _linha(id="2", nota_barato=Nota.ERRADA, nota_forte=Nota.CORRETA,
               tokens_resposta_barato=10, tokens_resposta_forte=100),
        _linha(id="3", nota_barato=Nota.PARCIAL, nota_forte=Nota.PARCIAL,
               tokens_resposta_barato=20, tokens_resposta_forte=80),
        _linha(id="4", nota_barato=Nota.CORRETA, nota_forte=Nota.PARCIAL,
               tokens_resposta_barato=10, tokens_resposta_forte=50),
    ]


def _por_nome(resultados):
    return {resultado.nome: resultado for resultado in resultados}


def test_avalia_politicas_com_resultado_calculado_a_mao():
    resultados = _por_nome(
        avaliar_politicas(_dataset(), _cfg(), tokenizador=TokenizadorFixo(10))
    )

    barato = resultados["sempre_barato"]
    forte = resultados["sempre_forte"]
    perfeito = resultados["perfeito"]

    assert barato.qualidade_media == pytest.approx(0.625)
    assert barato.custo_estimado_usd == pytest.approx(140.0)
    assert barato.qualidade_relativa_pct == pytest.approx(83.333333, rel=1e-5)
    assert barato.custo_relativo_pct == pytest.approx(5.0)
    assert barato.economia_pct == pytest.approx(95.0)

    assert forte.qualidade_media == pytest.approx(0.75)
    assert forte.custo_estimado_usd == pytest.approx(2800.0)
    assert forte.qualidade_relativa_pct == 100.0
    assert forte.custo_relativo_pct == 100.0
    assert forte.economia_pct == 0.0

    assert perfeito.qualidade_media == pytest.approx(0.875)
    assert perfeito.custo_estimado_usd == pytest.approx(950.0)
    assert perfeito.qualidade_relativa_pct == pytest.approx(116.666667, rel=1e-5)


def test_perfeito_nunca_perde_em_qualidade():
    resultados = avaliar_politicas(
        _dataset(), _cfg(), tokenizador=TokenizadorFixo(10)
    )
    perfeito = _por_nome(resultados)["perfeito"]
    disponiveis = [resultado for resultado in resultados if resultado.disponivel]

    assert all(
        perfeito.qualidade_media >= resultado.qualidade_media
        for resultado in disponiveis
    )


def test_sempre_forte_da_100_por_cento_por_definicao():
    forte = _por_nome(
        avaliar_politicas(_dataset(), _cfg(), tokenizador=TokenizadorFixo(10))
    )["sempre_forte"]

    assert forte.qualidade_relativa_pct == 100.0
    assert forte.custo_relativo_pct == 100.0
    assert forte.economia_pct == 0.0


def test_aleatorio_e_reprodutivel_com_a_mesma_semente():
    primeira = _por_nome(
        avaliar_politicas(
            _dataset(), _cfg(), tokenizador=TokenizadorFixo(10), semente=7
        )
    )["aleatorio"]
    segunda = _por_nome(
        avaliar_politicas(
            _dataset(), _cfg(), tokenizador=TokenizadorFixo(10), semente=7
        )
    )["aleatorio"]

    assert primeira == segunda


@pytest.mark.parametrize(("proporcao", "comparar"), [(0.0, "sempre_forte"), (1.0, "sempre_barato")])
def test_aleatorio_nas_bordas_iguala_as_politicas_fixas(proporcao, comparar):
    resultados = _por_nome(
        avaliar_politicas(
            _dataset(),
            _cfg(),
            tokenizador=TokenizadorFixo(10),
            proporcao=proporcao,
        )
    )

    assert resultados["aleatorio"].qualidade_media == pytest.approx(
        resultados[comparar].qualidade_media
    )
    assert resultados["aleatorio"].custo_estimado_usd == pytest.approx(
        resultados[comparar].custo_estimado_usd
    )


def test_proporcao_invalida_falha():
    with pytest.raises(ValueError, match="entre 0 e 1"):
        avaliar_politicas(_dataset(), _cfg(), proporcao=1.5)


def test_dataset_vazio_falha():
    with pytest.raises(ValueError, match="sem linhas"):
        avaliar_politicas([], _cfg())


def test_v1_vai_ao_forte_quando_o_contexto_nao_cabe():
    linha = _linha(
        pergunta="pergunta",
        nota_barato=Nota.ERRADA,
        nota_forte=Nota.CORRETA,
        contexto_tokens=200,
    )
    resultados = _por_nome(
        avaliar_politicas(
            [linha],
            _cfg(janela_contexto_barato=150),
            tokenizador=TokenizadorFixo(10),
        )
    )

    assert resultados["v1"].qualidade_media == pytest.approx(1.0)
    assert resultados["v1"].custo_estimado_usd == pytest.approx(
        resultados["sempre_forte"].custo_estimado_usd
    )


def test_v1_vai_ao_forte_quando_a_pergunta_passa_do_limiar():
    linha = _linha(nota_barato=Nota.ERRADA, nota_forte=Nota.CORRETA)
    resultados = _por_nome(
        avaliar_politicas(
            [linha],
            _cfg(limiar_tokens_pergunta=5),
            tokenizador=TokenizadorFixo(10),
        )
    )

    assert resultados["v1"].qualidade_media == pytest.approx(1.0)


def test_v1_fica_no_barato_dentro_do_limiar():
    linha = _linha(nota_barato=Nota.CORRETA, nota_forte=Nota.ERRADA)
    resultados = _por_nome(
        avaliar_politicas(
            [linha],
            _cfg(limiar_tokens_pergunta=20),
            tokenizador=TokenizadorFixo(10),
        )
    )

    assert resultados["v1"].qualidade_media == pytest.approx(1.0)
    assert resultados["v1"].custo_estimado_usd == pytest.approx(
        resultados["sempre_barato"].custo_estimado_usd
    )


def test_forte_limitado_usa_as_colunas_limitadas():
    linhas = [
        _linha(
            id=str(indice),
            nota_forte_limitado=nota,
            tokens_resposta_forte_limitado=tokens,
        )
        for indice, (nota, tokens) in enumerate(
            [
                (Nota.CORRETA, 50),
                (Nota.CORRETA, 50),
                (Nota.PARCIAL, 40),
                (Nota.PARCIAL, 25),
            ],
            start=1,
        )
    ]
    resultado = _por_nome(
        avaliar_politicas(linhas, _cfg(), tokenizador=TokenizadorFixo(10))
    )["forte_limitado"]

    assert resultado.disponivel
    assert resultado.qualidade_media == pytest.approx(0.75)
    assert resultado.custo_estimado_usd == pytest.approx(1480.0)
    assert resultado.economia_pct == pytest.approx(55.952381, rel=1e-5)


def test_forte_limitado_sem_colunas_e_ignorado_com_aviso():
    resultados = _por_nome(
        avaliar_politicas(_dataset(), _cfg(), tokenizador=TokenizadorFixo(10))
    )
    limitado = resultados["forte_limitado"]

    assert not limitado.disponivel
    assert limitado.aviso
    assert "ignorada" in limitado.aviso


def test_politicas_extras_sao_avaliadas():
    resultados = _por_nome(
        avaliar_politicas(
            _dataset(),
            _cfg(),
            tokenizador=TokenizadorFixo(10),
            extras={"custom": escolha_barato},
        )
    )

    assert resultados["custom"].qualidade_media == pytest.approx(0.625)


def test_resultado_indisponivel_nao_tem_metricas_validas():
    limitado = _por_nome(
        avaliar_politicas(_dataset(), _cfg(), tokenizador=TokenizadorFixo(10))
    )["forte_limitado"]

    assert isinstance(limitado, ResultadoPolitica)
    assert limitado.qualidade_media == 0.0
    assert limitado.custo_estimado_usd == 0.0
