import pytest

from roteador.config import Config
from roteador.limite import limite_por_faixa, montar_prompt, teto_rigido
from roteador.tipos import Faixa


def test_montar_prompt_texto_exato():
    assert (
        montar_prompt("Pergunta", 100)
        == "Pergunta Limite seus gastos a cerca de 100 tokens."
    )


def test_montar_prompt_vazio_nao_comeca_com_espaco():
    assert montar_prompt("", 100) == "Limite seus gastos a cerca de 100 tokens."


def test_montar_prompt_com_x_invalido_falha():
    with pytest.raises(ValueError, match="maior que zero"):
        montar_prompt("Pergunta", 0)

    with pytest.raises(ValueError, match="maior que zero"):
        montar_prompt("Pergunta", -5)


def test_teto_rigido_arredonda_para_cima():
    cfg = Config(fator_teto=1.5)

    assert teto_rigido(10, cfg) == 15
    assert teto_rigido(11, cfg) == 17
    assert teto_rigido(7, cfg) == 11


def test_teto_rigido_respeita_o_fator_configurado():
    assert teto_rigido(5, Config(fator_teto=2.0)) == 10


def test_teto_rigido_com_x_invalido_falha():
    with pytest.raises(ValueError, match="maior que zero"):
        teto_rigido(0, Config())


def test_faixas_curta_e_longa_nao_tem_limite():
    assert limite_por_faixa(Faixa.CURTA, (10, 20)) is None
    assert limite_por_faixa(Faixa.LONGA, (10, 20)) is None


def test_faixa_media_usa_o_corte_superior():
    assert limite_por_faixa(Faixa.MEDIA, (10, 20)) == 20
    assert limite_por_faixa(Faixa.MEDIA, [10, 25]) == 25


def test_faixa_media_com_corte_fracionario_usa_a_parte_inteira():
    assert limite_por_faixa(Faixa.MEDIA, (10, 20.9)) == 20


def test_faixa_media_com_cortes_insuficientes_falha():
    with pytest.raises(ValueError, match="dois cortes"):
        limite_por_faixa(Faixa.MEDIA, (10,))


def test_faixa_media_com_x_invalido_falha():
    with pytest.raises(ValueError, match="maior que zero"):
        limite_por_faixa(Faixa.MEDIA, (10, 0))
