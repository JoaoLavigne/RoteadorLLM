import pytest

from roteador.config import Config
from roteador.registro import COLUNAS_REGISTRO
from roteador.tokenizador import (
    ErroTokenizador,
    TokenizadorAproximado,
    TokenizadorComReserva,
    TokenizadorReal,
    criar_tokenizador,
)


def _tokenizador_real() -> TokenizadorReal:
    pytest.importorskip("tiktoken")
    tokenizador = TokenizadorReal("tiktoken:o200k_base")
    try:
        tokenizador.contar("teste")
    except ErroTokenizador as exc:
        pytest.skip(f"codificação indisponível: {exc}")
    return tokenizador


def test_texto_vazio_conta_zero():
    assert TokenizadorAproximado().contar("") == 0


def test_texto_curto_conta_um():
    assert TokenizadorAproximado().contar("abc") == 1


def test_texto_no_limite_exato():
    assert TokenizadorAproximado().contar("abcd") == 1
    assert TokenizadorAproximado().contar("abcdefgh") == 2


def test_texto_longo_arredonda_para_cima():
    assert TokenizadorAproximado().contar("a" * 9) == 3


def test_texto_com_acentos():
    assert TokenizadorAproximado().contar("olá") == 1
    assert TokenizadorAproximado().contar("coração") == 2


def test_divisor_configuravel():
    assert TokenizadorAproximado(divisor=2).contar("abcdefgh") == 4

    with pytest.raises(ValueError, match="maior que zero"):
        TokenizadorAproximado(divisor=0)


def test_satisfaz_o_protocolo_tokenizador():
    tokenizador = TokenizadorAproximado()

    assert callable(getattr(tokenizador, "contar", None))
    assert tokenizador.contar("teste") == 2


def test_real_mesmo_texto_mesma_contagem():
    tokenizador = _tokenizador_real()
    texto = "Explique a diferença entre treino e teste"

    assert tokenizador.contar(texto) == tokenizador.contar(texto)


def test_real_texto_vazio_conta_zero():
    assert _tokenizador_real().contar("") == 0


def test_real_carrega_apenas_no_primeiro_uso():
    pytest.importorskip("tiktoken")
    tokenizador = TokenizadorReal("tiktoken:o200k_base")
    assert tokenizador._codificador is None

    try:
        tokenizador.contar("teste")
    except ErroTokenizador as exc:
        pytest.skip(f"codificação indisponível: {exc}")

    assert tokenizador._codificador is not None


def test_real_fica_dentro_da_tolerancia_do_aproximado():
    real = _tokenizador_real()
    aproximado = TokenizadorAproximado()
    textos = [
        "Explique a diferença entre treino e teste",
        "Crie um CRUD completo em Python com autenticação e testes.",
        "O documento fala sobre o quê?",
        "Quais são as etapas do processo descrito no texto e como elas se relacionam?",
        "coração",
    ]

    for texto in textos:
        razao = real.contar(texto) / aproximado.contar(texto)
        assert 0.5 <= razao <= 2.0


def test_especificacao_invalida_falha():
    with pytest.raises(ErroTokenizador, match="inválida"):
        TokenizadorReal("o200k_base")


class _RealQuebrado:
    nome = "real"

    def __init__(self) -> None:
        self.chamadas = 0

    def contar(self, texto: str) -> int:
        self.chamadas += 1
        raise ErroTokenizador("falha simulada na carga")


def test_falha_simulada_na_carga_aciona_a_reserva():
    principal = _RealQuebrado()
    tokenizador = TokenizadorComReserva(principal, TokenizadorAproximado())

    assert tokenizador.contar("abcdefgh") == 2
    assert tokenizador.tokenizador_usado == "aproximado"

    assert tokenizador.contar("abcdefgh") == 2
    assert principal.chamadas == 1


def test_fabrica_monta_real_com_reserva():
    cfg = Config()
    tokenizador = criar_tokenizador(cfg)

    assert isinstance(tokenizador, TokenizadorComReserva)
    assert isinstance(tokenizador.principal, TokenizadorReal)
    assert isinstance(tokenizador.reserva, TokenizadorAproximado)
    assert tokenizador.reserva.divisor == cfg.divisor_aproximado


def test_registro_tem_coluna_tokenizador_usado():
    assert "tokenizador_usado" in COLUNAS_REGISTRO
