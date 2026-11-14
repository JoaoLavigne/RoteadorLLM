from roteador.avaliacao.relatorio import (
    RelatorioAlertas,
    TaxaAlerta,
    gerar_relatorio_alertas,
)
from roteador.config import Config


def _cfg(limiar=0.05):
    return Config(limiar_alerta=limiar)


def _registro(modelo, faixa):
    return {"modelo": modelo, "faixa_prevista": faixa}


def _ml(modelo, faixa, tipo="vazio"):
    return {
        "modelo": modelo,
        "faixa_prevista": faixa,
        "tipo_ocorrencia": tipo,
    }


def _taxa(taxas, valor):
    return next(taxa for taxa in taxas if taxa.valor == valor)


def test_calcula_taxa_por_modelo_e_por_faixa():
    registro = (
        [_registro("barato", "curta") for _ in range(5)]
        + [_registro("barato", "longa") for _ in range(5)]
        + [_registro("forte", "media") for _ in range(10)]
    )
    ml = [
        _ml("barato", "curta"),
        _ml("barato", "longa"),
    ]

    relatorio = gerar_relatorio_alertas(registro, ml, _cfg())

    assert relatorio.por_modelo == (
        TaxaAlerta("barato", 2, 10, 0.2, True),
        TaxaAlerta("forte", 0, 10, 0.0, False),
    )
    assert relatorio.por_faixa == (
        TaxaAlerta("curta", 1, 5, 0.2, True),
        TaxaAlerta("longa", 1, 5, 0.2, True),
        TaxaAlerta("media", 0, 10, 0.0, False),
    )
    assert len(relatorio.alertas) == 3


def test_ocorrencia_nenhuma_nao_entra_na_taxa():
    registro = [_registro("barato", "curta") for _ in range(10)]
    ml = [_ml("barato", "curta", tipo="nenhuma")]

    relatorio = gerar_relatorio_alertas(registro, ml, _cfg())

    assert _taxa(relatorio.por_modelo, "barato").taxa == 0.0
    assert _taxa(relatorio.por_faixa, "curta").taxa == 0.0


def test_limiar_e_aplicado_de_forma_estrita():
    registro = (
        [_registro("igual", "curta") for _ in range(20)]
        + [_registro("acima", "longa") for _ in range(19)]
    )
    ml = [_ml("igual", "curta"), _ml("acima", "longa")]

    relatorio = gerar_relatorio_alertas(registro, ml, _cfg(limiar=0.05))

    assert _taxa(relatorio.por_modelo, "igual").taxa == 0.05
    assert _taxa(relatorio.por_modelo, "igual").alerta is False
    assert _taxa(relatorio.por_modelo, "acima").taxa > 0.05
    assert _taxa(relatorio.por_modelo, "acima").alerta is True


def test_faixa_vazia_fica_fora_do_agrupamento_por_faixa():
    registro = [_registro("barato", "")]
    ml = [_ml("barato", "")]

    relatorio = gerar_relatorio_alertas(registro, ml, _cfg())

    assert [taxa.valor for taxa in relatorio.por_faixa] == []
    assert _taxa(relatorio.por_modelo, "barato").taxa == 1.0


def test_modelo_sem_requisicoes_aparece_com_taxa_zero():
    relatorio = gerar_relatorio_alertas([], [_ml("desconhecido", "curta")], _cfg())

    taxa = _taxa(relatorio.por_modelo, "desconhecido")
    assert taxa.requisicoes == 0
    assert taxa.ocorrencias == 1
    assert taxa.taxa == 0.0
    assert taxa.alerta is False


def test_relatorio_vazio_nao_quebra():
    relatorio = gerar_relatorio_alertas([], [], _cfg())

    assert isinstance(relatorio, RelatorioAlertas)
    assert relatorio.por_modelo == ()
    assert relatorio.por_faixa == ()
    assert relatorio.alertas == ()
