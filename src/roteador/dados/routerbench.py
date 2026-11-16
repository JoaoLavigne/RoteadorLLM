"""Leitura e conversão de recortes do RouterBench para o dataset do roteador.

O RouterBench guarda, para cada prompt, a nota, a resposta e o custo de vários
modelos. A calibração do v2 só precisa de um par barato/forte convertido para
    ``LinhaDataset``. O score bruto dos dois modelos é preservado na linha e é
    o que define "o barato basta"; as notas de três níveis ficam como resumo.
    O vínculo treino/validação/teste usa o próprio prompt como documento, para
    que o mesmo prompt nunca apareça em mais de um conjunto.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from roteador.dados.carregador import LinhaDataset, Nota
from roteador.tipos import Tokenizador

COLUNAS_FIXAS = ("sample_id", "prompt", "eval_name")
SUFIXO_RESPOSTA = "|model_response"
SUFIXO_CUSTO = "|total_cost"

DIVISOR_APROXIMADO = 4
EXTENSOES_PICKLE = (".pkl", ".pickle")
EXTENSOES_CSV = (".csv",)
LIMIAR_CORRETA = 1.0 - 1e-9


def carregar_dataframe(caminho: str | Path) -> pd.DataFrame:
    """Carrega o RouterBench de um pickle ou CSV."""
    arquivo = Path(caminho)
    if not arquivo.exists():
        raise FileNotFoundError(f"Arquivo do RouterBench não encontrado: {arquivo}")

    extensao = arquivo.suffix.lower()
    if extensao in EXTENSOES_PICKLE:
        dados = pd.read_pickle(arquivo)
    elif extensao in EXTENSOES_CSV:
        dados = pd.read_csv(arquivo)
    else:
        aceitas = ", ".join((*EXTENSOES_PICKLE, *EXTENSOES_CSV))
        raise ValueError(
            f"Extensão não suportada em {arquivo}; use {aceitas}."
        )

    if not isinstance(dados, pd.DataFrame):
        raise ValueError(f"O arquivo {arquivo} não contém uma tabela.")
    return dados


def modelos_disponiveis(dados: pd.DataFrame) -> list[str]:
    """Lista os modelos com nota, resposta e custo presentes na tabela."""
    modelos = []
    for coluna in dados.columns:
        if not isinstance(coluna, str):
            continue
        if coluna in COLUNAS_FIXAS or "|" in coluna:
            continue
        if (
            f"{coluna}{SUFIXO_RESPOSTA}" in dados.columns
            and f"{coluna}{SUFIXO_CUSTO}" in dados.columns
        ):
            modelos.append(coluna)
    return modelos


def normalizar_prompt(valor: Any) -> str:
    """Transforma o prompt do RouterBench em texto simples.

    No arquivo do RouterBench, o campo ``prompt`` é a representação em texto de
    uma lista de mensagens. Quando a conversão é possível, as mensagens são
    unidas por linhas em branco.
    """
    texto = "" if valor is None or pd.isna(valor) else str(valor).strip()
    if not texto.startswith("["):
        return texto

    try:
        partes = ast.literal_eval(texto)
    except (ValueError, SyntaxError):
        return texto

    if not isinstance(partes, (list, tuple)):
        return texto
    if not all(isinstance(parte, str) for parte in partes):
        return texto

    return "\n\n".join(parte.strip() for parte in partes if parte.strip())


def nota_de_score(score: float) -> Nota:
    """Converte a nota contínua do RouterBench para a escala do roteador.

    A nota cheia (1.0) vira correta, o zero vira errada e qualquer crédito
    parcial fica como parcial.
    """
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Score inválido: {score!r}.")
    if isinstance(score, float) and math.isnan(score):
        raise ValueError("Score inválido: NaN.")
    if score < 0:
        raise ValueError(f"Score inválido: {score!r}.")

    if score >= LIMIAR_CORRETA:
        return Nota.CORRETA
    if score > 0:
        return Nota.PARCIAL
    return Nota.ERRADA


def converter_para_dataset(
    dados: pd.DataFrame,
    barato: str,
    forte: str,
    tokenizador: Tokenizador | None = None,
    divisor_aproximado: int = DIVISOR_APROXIMADO,
) -> list[LinhaDataset]:
    """Converte as linhas de um par de modelos em ``LinhaDataset``.

    Linhas sem nota para o barato ou para o forte são ignoradas, porque não
    permitem decidir se o barato basta. O score bruto de cada modelo fica em
    ``score_barato`` e ``score_forte``. Sem tokenizador, os tokens de resposta
    são aproximados pelo número de caracteres dividido por ``divisor_aproximado``.
    """
    _validar_colunas(dados)
    _validar_modelo(dados, barato)
    _validar_modelo(dados, forte)
    if divisor_aproximado <= 0:
        raise ValueError("O divisor aproximado deve ser positivo.")

    linhas: list[LinhaDataset] = []
    for indice, registro in dados.iterrows():
        score_barato = registro[barato]
        score_forte = registro[forte]
        if pd.isna(score_barato) or pd.isna(score_forte):
            continue

        pergunta = normalizar_prompt(
            _texto_obrigatorio(registro, "prompt", indice)
        )
        if not pergunta:
            raise ValueError(f"Linha {indice}: o campo prompt está vazio.")
        linhas.append(
            LinhaDataset(
                id=_texto_obrigatorio(registro, "sample_id", indice),
                documento=pergunta,
                tipo=_texto_obrigatorio(registro, "eval_name", indice),
                pergunta=pergunta,
                nota_barato=nota_de_score(float(score_barato)),
                nota_forte=nota_de_score(float(score_forte)),
                tokens_resposta_barato=_contar_resposta(
                    registro[f"{barato}{SUFIXO_RESPOSTA}"],
                    tokenizador,
                    divisor_aproximado,
                ),
                tokens_resposta_forte=_contar_resposta(
                    registro[f"{forte}{SUFIXO_RESPOSTA}"],
                    tokenizador,
                    divisor_aproximado,
                ),
                score_barato=float(score_barato),
                score_forte=float(score_forte),
            )
        )

    if not linhas:
        raise ValueError("Nenhuma linha válida para o par de modelos informado.")
    return linhas


def selecionar_recorte(
    dados: pd.DataFrame, quantidade: int, semente: int = 42
) -> pd.DataFrame:
    """Sorteia um recorte reprodutível para versionar no projeto."""
    if quantidade <= 0:
        raise ValueError("A quantidade do recorte deve ser positiva.")
    if quantidade > len(dados):
        raise ValueError(
            f"A quantidade do recorte ({quantidade}) supera as "
            f"{len(dados)} linhas disponíveis."
        )
    return dados.sample(n=quantidade, random_state=semente).reset_index(drop=True)


def _validar_colunas(dados: pd.DataFrame) -> None:
    faltando = [coluna for coluna in COLUNAS_FIXAS if coluna not in dados.columns]
    if faltando:
        raise ValueError(
            "RouterBench sem as colunas obrigatórias: " + ", ".join(faltando) + "."
        )


def _validar_modelo(dados: pd.DataFrame, modelo: str) -> None:
    esperadas = (
        modelo,
        f"{modelo}{SUFIXO_RESPOSTA}",
        f"{modelo}{SUFIXO_CUSTO}",
    )
    faltando = [coluna for coluna in esperadas if coluna not in dados.columns]
    if faltando:
        disponiveis = ", ".join(modelos_disponiveis(dados)) or "nenhum"
        raise ValueError(
            f"Modelo {modelo!r} incompleto; faltando {', '.join(faltando)}. "
            f"Modelos disponíveis: {disponiveis}."
        )


def _texto_obrigatorio(registro: pd.Series, campo: str, indice: Any) -> str:
    valor = registro[campo]
    if valor is None or pd.isna(valor) or str(valor).strip() == "":
        raise ValueError(f"Linha {indice}: o campo {campo} está vazio.")
    return str(valor).strip()


def _contar_resposta(
    resposta: Any, tokenizador: Tokenizador | None, divisor_aproximado: int
) -> int:
    texto = "" if resposta is None or pd.isna(resposta) else str(resposta)
    if tokenizador is not None:
        return max(0, int(tokenizador.contar(texto)))
    return max(0, math.ceil(len(texto) / divisor_aproximado))


def escolher_par_mais_barato(
    dados: pd.DataFrame, meta_qualidade: float = 0.95
) -> tuple[str, str] | None:
    """Sugere o par barato/forte de menor custo que mantém a meta de qualidade.

    O forte é o modelo de maior qualidade média. O barato é o modelo mais
    barato cuja qualidade média chega a ``meta_qualidade`` da qualidade do
    forte. Devolve ``None`` se nenhum par atender à meta.
    """
    if not 0 < meta_qualidade <= 1:
        raise ValueError("A meta de qualidade deve estar entre 0 e 1.")

    _validar_colunas(dados)
    modelos = modelos_disponiveis(dados)
    if len(modelos) < 2:
        raise ValueError("O RouterBench precisa de pelo menos dois modelos.")

    qualidades = {
        modelo: float(dados[modelo].mean(skipna=True)) for modelo in modelos
    }
    for modelo, qualidade in qualidades.items():
        if math.isnan(qualidade):
            raise ValueError(f"O modelo {modelo!r} não tem notas válidas.")

    custos = {
        modelo: float(dados[f"{modelo}{SUFIXO_CUSTO}"].mean(skipna=True))
        for modelo in modelos
    }
    forte = max(
        modelos,
        key=lambda modelo: (qualidades[modelo], custos[modelo], modelo),
    )

    candidatos = [
        modelo
        for modelo in modelos
        if modelo != forte
        and qualidades[modelo] >= meta_qualidade * qualidades[forte]
    ]
    if not candidatos:
        return None

    barato = min(
        candidatos, key=lambda modelo: (custos[modelo], qualidades[modelo], modelo)
    )
    return barato, forte


def resumo_par(
    dados: pd.DataFrame, barato: str, forte: str
) -> dict[str, dict[str, float]]:
    """Resume qualidade e custo médios de um par, para conferência manual."""
    _validar_colunas(dados)
    _validar_modelo(dados, barato)
    _validar_modelo(dados, forte)

    def _resumo(modelo: str) -> dict[str, float]:
        return {
            "qualidade_media": float(dados[modelo].mean(skipna=True)),
            "custo_medio_usd": float(
                dados[f"{modelo}{SUFIXO_CUSTO}"].mean(skipna=True)
            ),
        }

    return {barato: _resumo(barato), forte: _resumo(forte)}


def colunas_de_modelo(dados: pd.DataFrame) -> Sequence[str]:
    """Devolve as colunas de nota dos modelos, na ordem da tabela."""
    return tuple(modelos_disponiveis(dados))
