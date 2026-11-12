from roteador.config import Config
from roteador.roteador_v1 import RoteadorV1
from roteador.tipos import Papel


class TokenizadorPorCaractere:
    def contar(self, texto: str) -> int:
        return len(texto)


def _roteador(**cfg_extra) -> RoteadorV1:
    valores = {
        "janela_contexto_barato": 100,
        "limiar_tokens_pergunta": 5,
        "x_tokens_forte_limitado": 50,
    }
    valores.update(cfg_extra)
    return RoteadorV1(TokenizadorPorCaractere(), Config(**valores))


def test_pergunta_abaixo_do_limiar_vai_ao_barato():
    decisao = _roteador().decidir("1234")

    assert decisao.papel == Papel.BARATO
    assert decisao.limite_tokens is None
    assert decisao.faixa_prevista is None
    assert decisao.motivo


def test_pergunta_igual_ao_limiar_vai_ao_barato():
    decisao = _roteador().decidir("12345")

    assert decisao.papel == Papel.BARATO


def test_pergunta_acima_do_limiar_vai_ao_forte():
    decisao = _roteador().decidir("123456")

    assert decisao.papel == Papel.FORTE
    assert decisao.motivo == "pergunta acima do limiar"
    assert decisao.limite_tokens == 50
    assert decisao.faixa_prevista is None


def test_contexto_grande_vai_ao_forte():
    decisao = _roteador(janela_contexto_barato=10).decidir("1234", contexto="1234567")

    assert decisao.papel == Papel.FORTE
    assert decisao.motivo == "contexto não cabe no barato"


def test_contexto_exatamente_no_limite_ainda_cabe():
    decisao = _roteador(janela_contexto_barato=10).decidir("1234", contexto="123456")

    assert decisao.papel == Papel.BARATO


def test_contexto_grande_tem_prioridade_sobre_o_limiar():
    decisao = _roteador(janela_contexto_barato=10).decidir("123456", contexto="12345")

    assert decisao.papel == Papel.FORTE
    assert decisao.motivo == "contexto não cabe no barato"


def test_limite_do_forte_vem_da_config():
    assert _roteador(x_tokens_forte_limitado=70).decidir("123456").limite_tokens == 70
    assert _roteador(x_tokens_forte_limitado=None).decidir("123456").limite_tokens is None


def test_barato_nunca_recebe_limite():
    assert _roteador().decidir("1234").limite_tokens is None


def test_duas_perguntas_com_mesma_contagem_recebem_a_mesma_decisao():
    roteador = _roteador()

    assert roteador.decidir("abcde") == roteador.decidir("12345")


def test_motivos_ficam_preenchidos():
    roteador = _roteador(janela_contexto_barato=10)

    assert roteador.decidir("1234").motivo
    assert roteador.decidir("123456").motivo
    assert roteador.decidir("1234", contexto="1234567").motivo


def test_satisfaz_o_protocolo_de_roteador():
    roteador = _roteador()

    assert callable(getattr(roteador, "decidir", None))
    assert roteador.decidir("1234").papel == Papel.BARATO
