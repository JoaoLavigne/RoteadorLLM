import pytest

from roteador.adapters.base import ChamadaAdapter, ErroAdapter
from roteador.adapters.falso import AdapterFalso
from roteador.tipos import MotivoParada, RespostaModelo


def _resposta(texto: str = "ok") -> RespostaModelo:
    return RespostaModelo(
        texto=texto,
        tokens_entrada=1,
        tokens_saida=1,
        tokens_raciocinio=None,
        motivo_parada=MotivoParada.COMPLETO,
        tempo_ms=1.0,
        modelo="falso",
    )


def test_sequencia_de_respostas_e_respeitada():
    primeira = _resposta("primeira")
    segunda = _resposta("segunda")
    adapter = AdapterFalso([primeira, segunda])

    assert adapter.gerar("p1", None) is primeira
    assert adapter.gerar("p2", 10) is segunda


def test_chamadas_ficam_registradas():
    adapter = AdapterFalso([_resposta()])

    adapter.gerar("pergunta", 42)

    assert adapter.chamadas == [ChamadaAdapter("pergunta", 42)]
    assert adapter.chamadas[0].prompt == "pergunta"
    assert adapter.chamadas[0].limite_tokens == 42


def test_limite_none_tambem_e_registrado():
    adapter = AdapterFalso([_resposta()])

    adapter.gerar("pergunta")

    assert adapter.chamadas[0].limite_tokens is None


def test_esgotar_as_respostas_levanta_erro_claro():
    adapter = AdapterFalso([_resposta()])
    adapter.gerar("primeira", None)

    with pytest.raises(ErroAdapter, match="sem respostas programadas") as exc:
        adapter.gerar("segunda", None)

    assert exc.value.transitorio is False
    assert len(adapter.chamadas) == 2


def test_lista_vazia_falha_na_primeira_chamada():
    adapter = AdapterFalso([])

    with pytest.raises(ErroAdapter, match="sem respostas programadas"):
        adapter.gerar("pergunta", None)


def test_satisfaz_o_protocolo_de_adapter():
    adapter = AdapterFalso([])

    assert isinstance(adapter.nome, str)
    assert callable(getattr(adapter, "gerar", None))


def test_nome_pode_ser_personalizado():
    adapter = AdapterFalso([], nome="falso-especial")

    assert adapter.nome == "falso-especial"
