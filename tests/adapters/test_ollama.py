import pytest

from roteador.adapters.base import ErroAdapter
from roteador.adapters.ollama import AdapterOllama
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


def _dados_ollama(**extra):
    dados = {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": "resposta do modelo"},
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 5,
    }
    dados.update(extra)
    return dados


def _adapter(post: PostFalso) -> AdapterOllama:
    return AdapterOllama(
        modelo="llama3.2:3b",
        host="http://localhost:11434",
        post=post,
    )


def test_resposta_normal():
    post = PostFalso(RespostaFalsa(_dados_ollama()))

    resposta = _adapter(post).gerar("pergunta")

    assert resposta.texto == "resposta do modelo"
    assert resposta.tokens_entrada == 12
    assert resposta.tokens_saida == 5
    assert resposta.tokens_raciocinio is None
    assert resposta.motivo_parada == MotivoParada.COMPLETO
    assert resposta.modelo == "llama3.2:3b"
    assert resposta.tempo_ms >= 0


def test_parada_por_limite_vira_truncado():
    post = PostFalso(RespostaFalsa(_dados_ollama(done_reason="length")))

    resposta = _adapter(post).gerar("pergunta", limite_tokens=10)

    assert resposta.motivo_parada == MotivoParada.TRUNCADO


def test_limite_vai_no_payload_como_num_predict():
    post = PostFalso(RespostaFalsa(_dados_ollama()))

    _adapter(post).gerar("pergunta", limite_tokens=30)

    url, kwargs = post.chamadas[0]
    assert url == "http://localhost:11434/api/chat"
    assert kwargs["json"]["options"] == {"num_predict": 30}
    assert kwargs["json"]["stream"] is False
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "pergunta"}]


def test_sem_limite_nao_envia_options():
    post = PostFalso(RespostaFalsa(_dados_ollama()))

    _adapter(post).gerar("pergunta")

    assert "options" not in post.chamadas[0][1]["json"]


def test_campos_ausentes_viram_vazio_sem_excecao():
    post = PostFalso(RespostaFalsa({}))

    resposta = _adapter(post).gerar("pergunta")

    assert resposta.texto == ""
    assert resposta.tokens_entrada == 0
    assert resposta.tokens_saida == 0
    assert resposta.motivo_parada == MotivoParada.VAZIO
    assert resposta.modelo == "llama3.2:3b"


def test_texto_com_caracteres_especiais_e_preservado():
    texto = 'ação, ção, "aspas", ponto e vírgula; barra | quebra\nlinha 2'
    post = PostFalso(
        RespostaFalsa(
            _dados_ollama(message={"role": "assistant", "content": texto})
        )
    )

    resposta = _adapter(post).gerar("pergunta")

    assert resposta.texto == texto


def test_erro_http_vira_erro_adapter():
    post = PostFalso(RespostaFalsa({}, status_code=500, texto="falha interna"))

    with pytest.raises(ErroAdapter, match="HTTP 500") as exc:
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
