import pytest

from roteador.adapters.base import ChamadaAdapter, ErroAdapter
from roteador.comunicacao import ExecutorComunicacao
from roteador.tipos import MotivoParada, Papel, RespostaModelo


def _resposta(modelo: str = "modelo") -> RespostaModelo:
    return RespostaModelo(
        texto="resposta.",
        tokens_entrada=10,
        tokens_saida=5,
        tokens_raciocinio=None,
        motivo_parada=MotivoParada.COMPLETO,
        tempo_ms=1.0,
        modelo=modelo,
    )


class AdapterRoteirizado:
    def __init__(self, nome: str, itens) -> None:
        self.nome = nome
        self._itens = list(itens)
        self._indice = 0
        self.chamadas: list[ChamadaAdapter] = []

    def gerar(
        self, prompt: str, limite_tokens: int | None = None
    ) -> RespostaModelo:
        self.chamadas.append(ChamadaAdapter(prompt, limite_tokens))
        if self._indice >= len(self._itens):
            raise ErroAdapter("lista acabou", transitorio=False)

        item = self._itens[self._indice]
        self._indice += 1
        if isinstance(item, Exception):
            raise item
        return item


class SleepFalso:
    def __init__(self) -> None:
        self.chamadas: list[float] = []

    def __call__(self, segundos: float) -> None:
        self.chamadas.append(segundos)


def _executor(adapters, **extra):
    opcoes = {
        "tentativas": 3,
        "espera": 0.0,
        "sleep": SleepFalso(),
        "fallback": False,
    }
    opcoes.update(extra)
    return ExecutorComunicacao(adapters, **opcoes)


def test_sucesso_na_primeira_nao_retenta_nem_faz_fallback():
    resposta = _resposta("barato")
    barato = AdapterRoteirizado("barato", [resposta])
    forte = AdapterRoteirizado("forte", [_resposta("forte")])
    sleep = SleepFalso()
    executor = _executor(
        {Papel.BARATO: barato, Papel.FORTE: forte},
        sleep=sleep,
        fallback=True,
    )

    resultado = executor.executar(Papel.BARATO, "base", "base limitada", None)

    assert resultado.resposta is resposta
    assert resultado.papel == Papel.BARATO
    assert resultado.prompt == "base"
    assert resultado.limite_aplicado is None
    assert resultado.tentativas == 1
    assert resultado.fallback is False
    assert resultado.erros == ()
    assert barato.chamadas == [ChamadaAdapter("base", None)]
    assert forte.chamadas == []
    assert sleep.chamadas == []


def test_retry_transitorio_ate_suceder():
    resposta = _resposta("barato")
    barato = AdapterRoteirizado(
        "barato", [ErroAdapter("rede caiu"), resposta]
    )
    sleep = SleepFalso()
    executor = _executor(
        {Papel.BARATO: barato},
        tentativas=3,
        espera=0.25,
        sleep=sleep,
    )

    resultado = executor.executar(Papel.BARATO, "base", "base limitada", None)

    assert resultado.resposta is resposta
    assert resultado.tentativas == 2
    assert resultado.fallback is False
    assert resultado.erros == ("barato: rede caiu",)
    assert len(barato.chamadas) == 2
    assert sleep.chamadas == [0.25]


def test_retry_esgota_e_falha_sem_fallback():
    barato = AdapterRoteirizado(
        "barato",
        [ErroAdapter("rede 1"), ErroAdapter("rede 2"), ErroAdapter("rede 3")],
    )
    sleep = SleepFalso()
    executor = _executor(
        {Papel.BARATO: barato},
        tentativas=3,
        espera=0.25,
        sleep=sleep,
    )

    with pytest.raises(ErroAdapter, match="barato falhou após 3") as exc:
        executor.executar(Papel.BARATO, "base", "base limitada", None)

    assert exc.value.transitorio is False
    assert len(barato.chamadas) == 3
    assert sleep.chamadas == [0.25, 0.25]


def test_erro_permanente_nao_retenta():
    barato = AdapterRoteirizado(
        "barato",
        [ErroAdapter("chave inválida", transitorio=False), _resposta()],
    )
    sleep = SleepFalso()
    executor = _executor(
        {Papel.BARATO: barato},
        tentativas=3,
        espera=0.25,
        sleep=sleep,
    )

    with pytest.raises(ErroAdapter, match="barato falhou após 1"):
        executor.executar(Papel.BARATO, "base", "base limitada", None)

    assert len(barato.chamadas) == 1
    assert sleep.chamadas == []


def test_fallback_para_outro_papel():
    resposta_forte = _resposta("forte")
    barato = AdapterRoteirizado(
        "barato", [ErroAdapter("rede 1"), ErroAdapter("rede 2")]
    )
    forte = AdapterRoteirizado("forte", [resposta_forte])
    executor = _executor(
        {Papel.BARATO: barato, Papel.FORTE: forte},
        tentativas=2,
        fallback=True,
    )

    resultado = executor.executar(Papel.BARATO, "base", "base limitada", None)

    assert resultado.resposta is resposta_forte
    assert resultado.papel == Papel.FORTE
    assert resultado.prompt == "base"
    assert resultado.limite_aplicado is None
    assert resultado.tentativas == 3
    assert resultado.fallback is True
    assert len(resultado.erros) == 2
    assert len(barato.chamadas) == 2
    assert forte.chamadas == [ChamadaAdapter("base", None)]


def test_fallback_do_forte_nao_leva_o_limite_para_o_barato():
    resposta_barato = _resposta("barato")
    forte = AdapterRoteirizado(
        "forte", [ErroAdapter("rede 1"), ErroAdapter("rede 2")]
    )
    barato = AdapterRoteirizado("barato", [resposta_barato])
    executor = _executor(
        {Papel.BARATO: barato, Papel.FORTE: forte},
        tentativas=2,
        fallback=True,
    )

    resultado = executor.executar(
        Papel.FORTE, "base", "base limitada", 150
    )

    assert resultado.resposta is resposta_barato
    assert resultado.papel == Papel.BARATO
    assert resultado.prompt == "base"
    assert resultado.limite_aplicado is None
    assert barato.chamadas == [ChamadaAdapter("base", None)]


def test_fallback_do_barato_para_o_forte_nao_aplica_limite():
    resposta_forte = _resposta("forte")
    barato = AdapterRoteirizado(
        "barato", [ErroAdapter("rede 1"), ErroAdapter("rede 2")]
    )
    forte = AdapterRoteirizado("forte", [resposta_forte])
    executor = _executor(
        {Papel.BARATO: barato, Papel.FORTE: forte},
        tentativas=2,
        fallback=True,
    )

    resultado = executor.executar(
        Papel.BARATO, "base", "base limitada", 150
    )

    assert resultado.resposta is resposta_forte
    assert resultado.papel == Papel.FORTE
    assert resultado.prompt == "base"
    assert resultado.limite_aplicado is None
    assert forte.chamadas == [ChamadaAdapter("base", None)]


def test_falha_nos_dois_papeis_levanta_erro_com_as_duas_mensagens():
    barato = AdapterRoteirizado(
        "barato", [ErroAdapter("falha barato 1"), ErroAdapter("falha barato 2")]
    )
    forte = AdapterRoteirizado(
        "forte", [ErroAdapter("falha forte 1"), ErroAdapter("falha forte 2")]
    )
    executor = _executor(
        {Papel.BARATO: barato, Papel.FORTE: forte},
        tentativas=2,
        fallback=True,
    )

    with pytest.raises(ErroAdapter) as exc:
        executor.executar(Papel.BARATO, "base", "base limitada", None)

    mensagem = str(exc.value)
    assert "barato falhou após 2" in mensagem
    assert "forte falhou após 2" in mensagem
    assert exc.value.transitorio is False


def test_tentativas_invalidas_falham_na_construcao():
    with pytest.raises(ValueError, match="pelo menos 1"):
        ExecutorComunicacao({}, tentativas=0)

    with pytest.raises(ValueError, match="negativa"):
        ExecutorComunicacao({}, espera=-0.1)
