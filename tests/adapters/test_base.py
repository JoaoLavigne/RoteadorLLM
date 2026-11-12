from roteador.adapters.base import (
    Adapter,
    ChamadaAdapter,
    ErroAdapter,
    inteiro_ou_zero,
    status_transitorio,
)
from roteador.tipos import Adapter as AdapterDoContrato


def test_erro_adapter_e_excecao_com_mensagem():
    erro = ErroAdapter("falhou de propósito")

    assert isinstance(erro, Exception)
    assert str(erro) == "falhou de propósito"
    assert erro.transitorio is True


def test_erro_adapter_pode_ser_permanente():
    erro = ErroAdapter("chave inválida", transitorio=False)

    assert erro.transitorio is False


def test_status_transitorio():
    assert status_transitorio(408) is True
    assert status_transitorio(425) is True
    assert status_transitorio(429) is True
    assert status_transitorio(500) is True
    assert status_transitorio(503) is True
    assert status_transitorio(400) is False
    assert status_transitorio(401) is False
    assert status_transitorio(404) is False


def test_base_reexporta_o_contrato_de_adapter():
    assert Adapter is AdapterDoContrato


def test_chamada_adapter_guarda_prompt_e_limite():
    chamada = ChamadaAdapter(prompt="pergunta", limite_tokens=10)

    assert chamada.prompt == "pergunta"
    assert chamada.limite_tokens == 10


def test_inteiro_ou_zero_normaliza_contagens():
    assert inteiro_ou_zero(5) == 5
    assert inteiro_ou_zero(3.9) == 3
    assert inteiro_ou_zero(-2) == 0
    assert inteiro_ou_zero(None) == 0
    assert inteiro_ou_zero("10") == 0
    assert inteiro_ou_zero(True) == 0
