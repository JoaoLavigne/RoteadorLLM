import pytest

from roteador.adapters.base import ErroAdapter
from roteador.adapters.gemini import AdapterGemini
from roteador.tipos import MotivoParada


class RespostaFalsa:
    def __init__(
        self,
        dados=None,
        status_code: int = 200,
        texto: str = "",
        erro_json: Exception | None = None,
    ) -> None:
        self._dados = dados
        self.status_code = status_code
        self.text = texto
        self._erro_json = erro_json

    def json(self):
        if self._erro_json is not None:
            raise self._erro_json
        return self._dados


class PostFalso:
    def __init__(self, resposta=None, erro: Exception | None = None) -> None:
        self.resposta = resposta
        self.erro = erro
        self.chamadas: list[tuple[str, dict]] = []

    def __call__(self, url: str, **kwargs):
        self.chamadas.append((url, kwargs))
        if self.erro is not None:
            raise self.erro
        return self.resposta


def _dados_gemini(**extra):
    dados = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "resposta do Gemini"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 3,
        },
    }
    dados.update(extra)
    return dados


def _adapter(post: PostFalso) -> AdapterGemini:
    return AdapterGemini(
        modelo="gemini-flash-teste",
        api_key="chave-teste",
        post=post,
    )


def test_resposta_normal():
    post = PostFalso(RespostaFalsa(_dados_gemini()))

    resposta = _adapter(post).gerar("pergunta")

    assert resposta.texto == "resposta do Gemini"
    assert resposta.tokens_entrada == 10
    assert resposta.tokens_saida == 20
    assert resposta.tokens_raciocinio == 3
    assert resposta.motivo_parada == MotivoParada.COMPLETO
    assert resposta.modelo == "gemini-flash-teste"
    assert resposta.tempo_ms >= 0


def test_parada_por_limite_vira_truncado():
    dados = _dados_gemini()
    dados["candidates"][0]["finishReason"] = "MAX_TOKENS"
    post = PostFalso(RespostaFalsa(dados))

    resposta = _adapter(post).gerar("pergunta", limite_tokens=10)

    assert resposta.motivo_parada == MotivoParada.TRUNCADO


def test_sem_parts_vira_vazio():
    dados = {
        "candidates": [
            {
                "content": {"role": "model"},
                "finishReason": "MAX_TOKENS",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0},
    }
    post = PostFalso(RespostaFalsa(dados))

    resposta = _adapter(post).gerar("pergunta", limite_tokens=10)

    assert resposta.texto == ""
    assert resposta.motivo_parada == MotivoParada.VAZIO


def test_tokens_de_raciocinio_ausentes_viram_none():
    dados = _dados_gemini()
    del dados["usageMetadata"]["thoughtsTokenCount"]
    post = PostFalso(RespostaFalsa(dados))

    resposta = _adapter(post).gerar("pergunta")

    assert resposta.tokens_raciocinio is None


def test_limite_vai_no_payload_como_max_output_tokens():
    post = PostFalso(RespostaFalsa(_dados_gemini()))

    _adapter(post).gerar("pergunta", limite_tokens=30)

    url, kwargs = post.chamadas[0]
    assert url == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-flash-teste:generateContent"
    )
    assert kwargs["params"] == {"key": "chave-teste"}
    assert kwargs["json"]["generationConfig"] == {"maxOutputTokens": 30}
    assert kwargs["json"]["contents"] == [
        {"role": "user", "parts": [{"text": "pergunta"}]}
    ]


def test_sem_limite_nao_envia_generation_config():
    post = PostFalso(RespostaFalsa(_dados_gemini()))

    _adapter(post).gerar("pergunta")

    assert "generationConfig" not in post.chamadas[0][1]["json"]


def test_campos_ausentes_viram_vazio_sem_excecao():
    post = PostFalso(RespostaFalsa({}))

    resposta = _adapter(post).gerar("pergunta")

    assert resposta.texto == ""
    assert resposta.tokens_entrada == 0
    assert resposta.tokens_saida == 0
    assert resposta.tokens_raciocinio is None
    assert resposta.motivo_parada == MotivoParada.VAZIO


def test_texto_com_caracteres_especiais_e_preservado():
    texto = 'ação, ção, "aspas", ponto e vírgula; barra | quebra\nlinha 2'
    dados = _dados_gemini()
    dados["candidates"][0]["content"]["parts"] = [{"text": texto}]
    post = PostFalso(RespostaFalsa(dados))

    resposta = _adapter(post).gerar("pergunta")

    assert resposta.texto == texto


def test_erro_http_vira_erro_adapter():
    post = PostFalso(RespostaFalsa({}, status_code=429, texto="cota excedida"))

    with pytest.raises(ErroAdapter, match="HTTP 429") as exc:
        _adapter(post).gerar("pergunta")

    assert exc.value.transitorio is True


def test_erro_http_400_nao_e_transitorio():
    post = PostFalso(RespostaFalsa({}, status_code=400, texto="pedido inválido"))

    with pytest.raises(ErroAdapter, match="HTTP 400") as exc:
        _adapter(post).gerar("pergunta")

    assert exc.value.transitorio is False


def test_erro_de_rede_vira_erro_adapter():
    post = PostFalso(erro=ConnectionError("sem rede"))

    with pytest.raises(ErroAdapter, match="Falha de rede"):
        _adapter(post).gerar("pergunta")


def test_json_invalido_vira_erro_adapter():
    post = PostFalso(RespostaFalsa(erro_json=ValueError("json quebrado")))

    with pytest.raises(ErroAdapter, match="JSON válido"):
        _adapter(post).gerar("pergunta")


def test_variaveis_de_ambiente_ausentes_falham(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ErroAdapter, match="GEMINI_MODEL"):
        AdapterGemini(post=PostFalso(RespostaFalsa({})))
