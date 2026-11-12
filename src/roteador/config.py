"""Configuração central do roteador e leitura de sobrescritas em JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Config:
    modelo_barato: str = "llama3.2:3b"
    janela_contexto_barato: int = 8192
    limiar_tokens_pergunta: int = 200
    tokenizador: str = "tiktoken:o200k_base"
    divisor_aproximado: int = 4
    x_tokens_forte_limitado: int | None = None
    fator_teto: float = 1.5
    fator_nova_tentativa: float = 2.0
    tentativas_comunicacao: int = 3
    espera_entre_tentativas_s: float = 0.5
    fallback_comunicacao: bool = True
    limiar_raciocinio: float = 0.9
    limiar_alerta: float = 0.05
    percentis_faixas: tuple[float, float] = (1 / 3, 2 / 3)
    limiar_probabilidade: float = 0.5
    embedder_modelo: str = "nomic-embed-text"
    preco_ref: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"barato": (0.10, 0.50), "forte": (2.00, 10.00)}
    )
    caminho_registro: str = "resultados/registro.csv"
    caminho_ml: str = "resultados/ml_ocorrencias.csv"
    caminho_dataset: str = "dados/dataset.csv"


def carregar_config(caminho: str | Path | None = None) -> Config:
    """Devolve a configuração padrão, sobrescrita pelo JSON informado."""
    if caminho is None:
        return Config()

    arquivo = Path(caminho)
    try:
        conteudo = arquivo.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {arquivo}"
        ) from None

    try:
        dados = json.loads(conteudo)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido em {arquivo}: {exc}") from exc

    if not isinstance(dados, dict):
        raise ValueError(
            f"O arquivo de configuração deve conter um objeto JSON: {arquivo}"
        )

    nomes_validos = {campo.name for campo in fields(Config)}
    desconhecidas = sorted(set(dados) - nomes_validos)
    if desconhecidas:
        validas = ", ".join(sorted(nomes_validos))
        raise ValueError(
            f"Chave(s) desconhecida(s) na configuração: {', '.join(desconhecidas)}. "
            f"Chaves válidas: {validas}."
        )

    return Config(**_normalizar(dados))


def _normalizar(dados: dict[str, Any]) -> dict[str, Any]:
    """Converte listas do JSON para as tuplas usadas pela Config."""
    valores = dict(dados)

    if "percentis_faixas" in valores:
        valores["percentis_faixas"] = tuple(valores["percentis_faixas"])

    if "preco_ref" in valores:
        precos = valores["preco_ref"]
        if not isinstance(precos, dict):
            raise ValueError("preco_ref deve ser um objeto JSON com os papéis.")
        valores["preco_ref"] = {
            papel: tuple(preco) for papel, preco in precos.items()
        }

    return valores
