import pytest

from roteador.adapters.base import ChamadaAdapter, ErroAdapter
from roteador.adapters.falso import AdapterFalso
from roteador.config import Config
from roteador.orquestrador import Orquestrador
from roteador.registro import (
    COLUNAS_ML,
    COLUNAS_REGISTRO,
    RegistroCsv,
    RegistroMemoria,
)
from roteador.tipos import (
    Decisao,
    MotivoParada,
    Papel,
    RespostaModelo,
    TipoOcorrencia,
)


class TokenizadorFalso:
    nome = "real"
    tokenizador_usado = "real"

    def contar(self, texto: str) -> int:
        return len(texto)


class RoteadorFalso:
    def __init__(self, decisao: Decisao) -> None:
        self.decisao = decisao
        self.chamadas: list[tuple[str, str]] = []

    def decidir(self, pergunta: str, contexto: str = "") -> Decisao:
        self.chamadas.append((pergunta, contexto))
        return self.decisao


class AdapterInstavel:
    def __init__(self, nome: str, itens) -> None:
        self.nome = nome
        self._itens = list(itens)
        self._indice = 0
        self.chamadas: list[ChamadaAdapter] = []

    def gerar(
        self, prompt: str, limite_tokens: int | None = None
    ) -> RespostaModelo:
        self.chamadas.append(ChamadaAdapter(prompt, limite_tokens))
        item = self._itens[self._indice]
        self._indice += 1
        if isinstance(item, Exception):
            raise item
        return item


def _resposta(
    texto: str = "resposta.",
    tokens_entrada: int = 10,
    tokens_saida: int = 5,
    tokens_raciocinio: int | None = None,
    motivo_parada: MotivoParada = MotivoParada.COMPLETO,
    modelo: str = "modelo",
) -> RespostaModelo:
    return RespostaModelo(
        texto=texto,
        tokens_entrada=tokens_entrada,
        tokens_saida=tokens_saida,
        tokens_raciocinio=tokens_raciocinio,
        motivo_parada=motivo_parada,
        tempo_ms=1.0,
        modelo=modelo,
    )


def _decisao(
    papel: Papel = Papel.BARATO,
    limite_tokens: int | None = None,
    motivo: str = "decisão de teste",
) -> Decisao:
    return Decisao(papel=papel, motivo=motivo, limite_tokens=limite_tokens)


def _cfg(**extra) -> Config:
    valores = {
        "x_tokens_forte_limitado": 100,
        "fator_teto": 1.5,
        "fator_nova_tentativa": 2.0,
        "tentativas_comunicacao": 1,
        "espera_entre_tentativas_s": 0.0,
        "fallback_comunicacao": False,
        "preco_ref": {"barato": (1.0, 2.0), "forte": (10.0, 20.0)},
    }
    valores.update(extra)
    return Config(**valores)


def _orc(tmp_path, decisao, respostas_barato, respostas_forte, cfg=None):
    barato = AdapterFalso(respostas_barato, nome="barato")
    forte = AdapterFalso(respostas_forte, nome="forte")
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    registro_ml = RegistroCsv(tmp_path / "ml.csv", COLUNAS_ML)
    orquestrador = Orquestrador(
        RoteadorFalso(decisao),
        {Papel.BARATO: barato, Papel.FORTE: forte},
        TokenizadorFalso(),
        registro,
        registro_ml,
        cfg or _cfg(),
    )
    return orquestrador, barato, forte, registro, registro_ml


def test_fluxo_normal_grava_registro_e_nao_grava_ml(tmp_path):
    decisao = _decisao()
    resposta = _resposta(tokens_entrada=1000, tokens_saida=500, modelo="barato-modelo")
    orquestrador, barato, forte, registro, registro_ml = _orc(
        tmp_path, decisao, [resposta], []
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.resposta is resposta
    assert resultado.decisao is decisao
    assert resultado.ocorrencia.tipo == TipoOcorrencia.NENHUMA
    assert resultado.tentativas == 1
    assert resultado.custo_estimado_usd == pytest.approx((1000 * 1 + 500 * 2) / 1_000_000)
    assert barato.chamadas == [ChamadaAdapter("pergunta", None)]
    assert forte.chamadas == []

    linhas = registro.ler()
    assert len(linhas) == 1
    linha = linhas[0]
    assert linha["papel"] == "barato"
    assert linha["modelo"] == "barato-modelo"
    assert linha["tokens_pergunta"] == str(len("pergunta"))
    assert linha["tokenizador_usado"] == "real"
    assert linha["tentativas"] == "1"
    assert linha["ocorrencia"] == "nenhuma"
    assert linha["limite_tokens"] == ""
    assert linha["tokens_raciocinio"] == ""
    assert registro_ml.ler() == []


def test_custo_inclui_tokens_de_raciocinio(tmp_path):
    decisao = _decisao(Papel.FORTE, limite_tokens=1_000_000)
    resposta = _resposta(
        tokens_entrada=1_000_000,
        tokens_saida=1_000_000,
        tokens_raciocinio=500_000,
        modelo="forte-modelo",
    )
    orquestrador, barato, forte, registro, _ = _orc(
        tmp_path, decisao, [], [resposta]
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.custo_estimado_usd == pytest.approx(10 + 20 + 10)
    assert barato.chamadas == []
    assert forte.chamadas[0].limite_tokens == 1_500_000
    assert forte.chamadas[0].prompt == (
        "pergunta Limite seus gastos a cerca de 1000000 tokens."
    )


def test_barato_vazio_escala_ao_forte(tmp_path):
    decisao = _decisao()
    vazia = _resposta(texto="", tokens_entrada=10, tokens_saida=5, modelo="barato-modelo")
    normal = _resposta(texto="resposta.", tokens_entrada=20, tokens_saida=7, modelo="forte-modelo")
    orquestrador, barato, forte, registro, registro_ml = _orc(
        tmp_path, decisao, [vazia], [normal]
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.tentativas == 2
    assert resultado.resposta is normal
    assert resultado.decisao is decisao
    assert resultado.ocorrencia.tipo == TipoOcorrencia.NENHUMA
    assert resultado.custo_estimado_usd == pytest.approx(
        (10 * 1 + 5 * 2) / 1_000_000 + (20 * 10 + 7 * 20) / 1_000_000
    )
    assert barato.chamadas == [ChamadaAdapter("pergunta", None)]
    assert forte.chamadas == [ChamadaAdapter("pergunta", None)]

    linhas_ml = registro_ml.ler()
    assert len(linhas_ml) == 1
    assert linhas_ml[0]["tipo_ocorrencia"] == "vazio"
    assert linhas_ml[0]["acao"] == "escalou_ao_forte"
    assert linhas_ml[0]["resultado_final"] == "ok"
    assert linhas_ml[0]["modelo"] == "barato-modelo"

    linha = registro.ler()[0]
    assert linha["papel"] == "barato"
    assert linha["modelo"] == "forte-modelo"
    assert linha["tentativas"] == "2"
    assert linha["ocorrencia"] == "nenhuma"
    assert linha["tokens_entrada"] == "30"
    assert linha["tokens_saida"] == "12"
    assert linha["id_requisicao"] == linhas_ml[0]["id_requisicao"]


def test_forte_truncado_repete_com_teto_dobrado(tmp_path):
    decisao = _decisao(Papel.FORTE, limite_tokens=100)
    truncada = _resposta(
        texto="resposta cortada",
        tokens_saida=50,
        motivo_parada=MotivoParada.TRUNCADO,
        modelo="forte-modelo",
    )
    normal = _resposta(texto="resposta.", tokens_saida=20, modelo="forte-modelo")
    orquestrador, barato, forte, registro, registro_ml = _orc(
        tmp_path, decisao, [], [truncada, normal]
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.tentativas == 2
    assert resultado.resposta is normal
    assert resultado.ocorrencia.tipo == TipoOcorrencia.NENHUMA
    assert barato.chamadas == []
    assert len(forte.chamadas) == 2
    assert forte.chamadas[0].prompt == (
        "pergunta Limite seus gastos a cerca de 100 tokens."
    )
    assert forte.chamadas[0].limite_tokens == 150
    assert forte.chamadas[1].prompt == forte.chamadas[0].prompt
    assert forte.chamadas[1].limite_tokens == 300

    linhas_ml = registro_ml.ler()
    assert len(linhas_ml) == 1
    assert linhas_ml[0]["acao"] == "nova_tentativa_teto_maior"
    assert linhas_ml[0]["limite_aplicado"] == "150"
    assert linhas_ml[0]["resultado_final"] == "ok"

    linha = registro.ler()[0]
    assert linha["limite_tokens"] == "100"
    assert linha["tentativas"] == "2"


def test_segunda_tentativa_tambem_falha_sem_terceira(tmp_path):
    decisao = _decisao(Papel.FORTE, limite_tokens=100)
    truncada = _resposta(
        texto="resposta cortada",
        tokens_saida=50,
        motivo_parada=MotivoParada.TRUNCADO,
        modelo="forte-modelo",
    )
    vazia = _resposta(texto="", tokens_saida=5, modelo="forte-modelo")
    orquestrador, _, forte, registro, registro_ml = _orc(
        tmp_path, decisao, [], [truncada, vazia]
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.tentativas == 2
    assert resultado.ocorrencia.tipo == TipoOcorrencia.VAZIO
    assert len(forte.chamadas) == 2

    linhas_ml = registro_ml.ler()
    assert len(linhas_ml) == 2
    assert linhas_ml[0]["acao"] == "nova_tentativa_teto_maior"
    assert linhas_ml[0]["resultado_final"] == "ainda_com_ocorrencia"
    assert linhas_ml[1]["acao"] == "nenhuma"
    assert linhas_ml[1]["limite_aplicado"] == "300"
    assert linhas_ml[1]["resultado_final"] == "ainda_com_ocorrencia"

    linha = registro.ler()[0]
    assert linha["ocorrencia"] == "vazio"
    assert linha["tentativas"] == "2"


def test_possivelmente_truncado_so_gera_alerta(tmp_path):
    decisao = _decisao(Papel.FORTE, limite_tokens=100)
    resposta = _resposta(
        texto="sem ponto final",
        tokens_saida=143,
        modelo="forte-modelo",
    )
    orquestrador, _, forte, registro, registro_ml = _orc(
        tmp_path, decisao, [], [resposta]
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.ocorrencia.tipo == TipoOcorrencia.POSSIVELMENTE_TRUNCADO
    assert resultado.tentativas == 1
    assert len(forte.chamadas) == 1

    linhas_ml = registro_ml.ler()
    assert len(linhas_ml) == 1
    assert linhas_ml[0]["acao"] == "nenhuma"
    assert linhas_ml[0]["resultado_final"] == "ainda_com_ocorrencia"
    assert registro.ler()[0]["ocorrencia"] == "possivelmente_truncado"


def test_x_e_ignorado_no_barato(tmp_path):
    decisao = _decisao(Papel.BARATO, limite_tokens=100)
    resposta = _resposta(modelo="barato-modelo")
    orquestrador, barato, forte, _, _ = _orc(tmp_path, decisao, [resposta], [])

    resultado = orquestrador.responder("pergunta")

    assert resultado.ocorrencia.tipo == TipoOcorrencia.NENHUMA
    assert barato.chamadas[0].prompt == "pergunta"
    assert barato.chamadas[0].limite_tokens is None
    assert forte.chamadas == []


def test_contexto_entra_no_prompt_mas_nao_em_tokens_pergunta(tmp_path):
    decisao = _decisao()
    resposta = _resposta(modelo="barato-modelo")
    orquestrador, barato, _, registro, _ = _orc(
        tmp_path, decisao, [resposta], []
    )

    orquestrador.responder("pergunta", "contexto")

    assert barato.chamadas[0].prompt == "contexto\n\npergunta"
    assert registro.ler()[0]["tokens_pergunta"] == str(len("pergunta"))


def test_orquestrador_aceita_outro_adapter_de_registro():
    decisao = _decisao()
    resposta = _resposta(modelo="barato-modelo")
    barato = AdapterFalso([resposta], nome="barato")
    forte = AdapterFalso([], nome="forte")
    registro = RegistroMemoria(COLUNAS_REGISTRO)
    registro_ml = RegistroMemoria(COLUNAS_ML)
    orquestrador = Orquestrador(
        RoteadorFalso(decisao),
        {Papel.BARATO: barato, Papel.FORTE: forte},
        TokenizadorFalso(),
        registro,
        registro_ml,
        _cfg(),
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.tentativas == 1
    assert len(registro.ler()) == 1
    assert registro_ml.ler() == []


def test_orquestrador_retenta_comunicacao_antes_de_escalar(tmp_path):
    decisao = _decisao()
    resposta = _resposta(modelo="barato-modelo")
    barato = AdapterInstavel("barato", [ErroAdapter("rede caiu"), resposta])
    forte = AdapterFalso([], nome="forte")
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    registro_ml = RegistroCsv(tmp_path / "ml.csv", COLUNAS_ML)
    orquestrador = Orquestrador(
        RoteadorFalso(decisao),
        {Papel.BARATO: barato, Papel.FORTE: forte},
        TokenizadorFalso(),
        registro,
        registro_ml,
        _cfg(tentativas_comunicacao=2, fallback_comunicacao=False),
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.tentativas == 2
    assert resultado.resposta is resposta
    assert len(barato.chamadas) == 2
    assert forte.chamadas == []
    assert registro.ler()[0]["tentativas"] == "2"
    assert registro.ler()[0]["modelo"] == "barato-modelo"
    assert registro_ml.ler() == []


def test_orquestrador_faz_fallback_de_comunicacao_para_o_forte(tmp_path):
    decisao = _decisao()
    resposta_forte = _resposta(
        tokens_entrada=20, tokens_saida=7, modelo="forte-modelo"
    )
    barato = AdapterInstavel(
        "barato", [ErroAdapter("rede 1"), ErroAdapter("rede 2")]
    )
    forte = AdapterFalso([resposta_forte], nome="forte")
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    registro_ml = RegistroCsv(tmp_path / "ml.csv", COLUNAS_ML)
    orquestrador = Orquestrador(
        RoteadorFalso(decisao),
        {Papel.BARATO: barato, Papel.FORTE: forte},
        TokenizadorFalso(),
        registro,
        registro_ml,
        _cfg(tentativas_comunicacao=2, fallback_comunicacao=True),
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.tentativas == 3
    assert resultado.resposta is resposta_forte
    assert resultado.decisao.papel == Papel.BARATO
    assert len(barato.chamadas) == 2
    assert forte.chamadas == [ChamadaAdapter("pergunta", None)]
    assert resultado.custo_estimado_usd == pytest.approx(
        (20 * 10 + 7 * 20) / 1_000_000
    )

    linha = registro.ler()[0]
    assert linha["papel"] == "barato"
    assert linha["modelo"] == "forte-modelo"
    assert linha["tentativas"] == "3"
    assert registro_ml.ler() == []


def test_orquestrador_propaga_erro_quando_os_dois_falham(tmp_path):
    decisao = _decisao()
    barato = AdapterInstavel("barato", [ErroAdapter("rede barato")])
    forte = AdapterInstavel("forte", [ErroAdapter("rede forte")])
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    registro_ml = RegistroCsv(tmp_path / "ml.csv", COLUNAS_ML)
    orquestrador = Orquestrador(
        RoteadorFalso(decisao),
        {Papel.BARATO: barato, Papel.FORTE: forte},
        TokenizadorFalso(),
        registro,
        registro_ml,
        _cfg(tentativas_comunicacao=1, fallback_comunicacao=True),
    )

    with pytest.raises(ErroAdapter, match="dois papéis"):
        orquestrador.responder("pergunta")

    assert registro.ler() == []
    assert registro_ml.ler() == []


def test_fallback_do_forte_nao_classifica_o_barato_como_truncado(tmp_path):
    decisao = _decisao(Papel.FORTE, limite_tokens=100)
    resposta_barato = _resposta(
        texto="resposta sem ponto final",
        tokens_saida=143,
        modelo="barato-modelo",
    )
    forte = AdapterInstavel("forte", [ErroAdapter("rede forte")])
    barato = AdapterFalso([resposta_barato], nome="barato")
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    registro_ml = RegistroCsv(tmp_path / "ml.csv", COLUNAS_ML)
    orquestrador = Orquestrador(
        RoteadorFalso(decisao),
        {Papel.BARATO: barato, Papel.FORTE: forte},
        TokenizadorFalso(),
        registro,
        registro_ml,
        _cfg(tentativas_comunicacao=1, fallback_comunicacao=True),
    )

    resultado = orquestrador.responder("pergunta")

    assert resultado.resposta is resposta_barato
    assert resultado.tentativas == 2
    assert resultado.ocorrencia.tipo == TipoOcorrencia.NENHUMA
    assert barato.chamadas == [ChamadaAdapter("pergunta", None)]
    assert registro.ler()[0]["papel"] == "forte"
    assert registro.ler()[0]["modelo"] == "barato-modelo"
    assert registro.ler()[0]["limite_tokens"] == "100"
    assert registro_ml.ler() == []


def test_registro_preserva_caracteres_especiais_do_modelo_e_motivo(tmp_path):
    especial = 'modelo "x", ação; barra | quebra\nlinha 2 \\ fim 😀'
    decisao = _decisao(motivo=especial)
    resposta = _resposta(modelo=especial)
    orquestrador, _, _, registro, _ = _orc(
        tmp_path, decisao, [resposta], []
    )

    orquestrador.responder("pergunta")

    linha = registro.ler()[0]
    assert list(linha) == list(COLUNAS_REGISTRO)
    assert linha["modelo"] == especial
    assert linha["motivo_decisao"] == especial


def test_registro_grava_vazios_e_nulos_como_texto_vazio(tmp_path):
    decisao = _decisao()
    resposta = _resposta(tokens_raciocinio=None, modelo="barato-modelo")
    orquestrador, _, _, registro, _ = _orc(
        tmp_path, decisao, [resposta], []
    )

    orquestrador.responder("pergunta")

    linha = registro.ler()[0]
    assert linha["faixa_prevista"] == ""
    assert linha["limite_tokens"] == ""
    assert linha["tokens_raciocinio"] == ""
    assert linha["ocorrencia"] == "nenhuma"
