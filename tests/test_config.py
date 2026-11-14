import json

import pytest

from roteador.config import Config, carregar_config


def test_padroes_corretos():
    cfg = Config()

    assert cfg.modelo_barato == "llama3.2:3b"
    assert cfg.janela_contexto_barato == 8192
    assert cfg.limiar_tokens_pergunta == 200
    assert cfg.tokenizador == "tiktoken:o200k_base"
    assert cfg.divisor_aproximado == 4
    assert cfg.x_tokens_forte_limitado is None
    assert cfg.fator_teto == 1.5
    assert cfg.fator_nova_tentativa == 2.0
    assert cfg.tentativas_comunicacao == 3
    assert cfg.espera_entre_tentativas_s == 0.5
    assert cfg.fallback_comunicacao is True
    assert cfg.limiar_raciocinio == 0.9
    assert cfg.limiar_alerta == 0.05
    assert cfg.percentis_faixas == (1 / 3, 2 / 3)
    assert cfg.limiar_probabilidade == 0.5
    assert cfg.embedder_modelo == "nomic-embed-text"
    assert cfg.preco_ref == {"barato": (0.10, 0.50), "forte": (2.00, 10.00)}
    assert cfg.caminho_registro == "resultados/registro.csv"
    assert cfg.caminho_ml == "resultados/ml_ocorrencias.csv"
    assert cfg.caminho_dataset == "dados/dataset.csv"
    assert cfg.caminho_modelo_v2 == "resultados/modelo_v2.joblib"


def test_sem_caminho_usa_padroes():
    assert carregar_config() == Config()


def test_json_sobrescreve_padroes(tmp_path):
    arquivo = tmp_path / "config.json"
    arquivo.write_text(
        json.dumps(
            {
                "modelo_barato": "outro-modelo",
                "limiar_tokens_pergunta": 42,
                "tokenizador": "tiktoken:cl100k_base",
                "divisor_aproximado": 5,
                "x_tokens_forte_limitado": 300,
                "tentativas_comunicacao": 5,
                "espera_entre_tentativas_s": 0.1,
                "fallback_comunicacao": False,
                "percentis_faixas": [0.5, 0.8],
                "preco_ref": {"barato": [0.2, 0.4], "forte": [3.0, 12.0]},
            }
        ),
        encoding="utf-8",
    )

    cfg = carregar_config(arquivo)

    assert cfg.modelo_barato == "outro-modelo"
    assert cfg.limiar_tokens_pergunta == 42
    assert cfg.tokenizador == "tiktoken:cl100k_base"
    assert cfg.divisor_aproximado == 5
    assert cfg.x_tokens_forte_limitado == 300
    assert cfg.tentativas_comunicacao == 5
    assert cfg.espera_entre_tentativas_s == 0.1
    assert cfg.fallback_comunicacao is False
    assert cfg.percentis_faixas == (0.5, 0.8)
    assert cfg.preco_ref == {"barato": (0.2, 0.4), "forte": (3.0, 12.0)}
    assert cfg.fator_teto == 1.5


def test_chave_desconhecida_falha(tmp_path):
    arquivo = tmp_path / "config.json"
    arquivo.write_text(json.dumps({"chave_invalida": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="desconhecida"):
        carregar_config(arquivo)


def test_arquivo_inexistente_falha(tmp_path):
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        carregar_config(tmp_path / "nao_existe.json")


def test_json_invalido_falha(tmp_path):
    arquivo = tmp_path / "config.json"
    arquivo.write_text("{invalido", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON inválido"):
        carregar_config(arquivo)


def test_json_precisa_ser_objeto(tmp_path):
    arquivo = tmp_path / "config.json"
    arquivo.write_text(json.dumps([1, 2]), encoding="utf-8")

    with pytest.raises(ValueError, match="objeto JSON"):
        carregar_config(arquivo)
