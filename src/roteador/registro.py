"""Interface de registro e adapters de armazenamento (CSV e memória)."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

COLUNAS_REGISTRO = (
    "id_requisicao",
    "timestamp",
    "papel",
    "modelo",
    "tokens_pergunta",
    "tokenizador_usado",
    "tokens_entrada",
    "tokens_saida",
    "tokens_raciocinio",
    "tempo_ms",
    "custo_estimado_usd",
    "motivo_decisao",
    "faixa_prevista",
    "limite_tokens",
    "motivo_parada",
    "ocorrencia",
    "tentativas",
)

COLUNAS_ML = (
    "id_requisicao",
    "timestamp",
    "modelo",
    "faixa_prevista",
    "limite_aplicado",
    "tokens_saida_reais",
    "tokens_raciocinio",
    "motivo_parada",
    "tipo_ocorrencia",
    "acao",
    "resultado_final",
)


@runtime_checkable
class Registro(Protocol):
    """Interface que qualquer adapter de armazenamento precisa cumprir."""

    def adicionar(self, linha: Mapping[str, Any]) -> None:
        """Grava uma linha no armazenamento."""
        ...

    def ler(self) -> list[dict[str, str]]:
        """Devolve as linhas gravadas, com valores convertidos para texto."""
        ...


class RegistroCsv:
    """Adapter de armazenamento em CSV; grava e fecha o arquivo a cada linha."""

    def __init__(self, caminho: str | Path, colunas: Sequence[str]) -> None:
        self.caminho = Path(caminho)
        self.colunas = tuple(colunas)
        self._garantir_cabecalho()

    def adicionar(self, linha: Mapping[str, Any]) -> None:
        _validar_linha(linha, self.colunas)

        with self.caminho.open("a", encoding="utf-8", newline="") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow([linha[coluna] for coluna in self.colunas])

    def ler(self) -> list[dict[str, str]]:
        return ler_csv(self.caminho)

    def _garantir_cabecalho(self) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        if self.caminho.exists() and self.caminho.stat().st_size > 0:
            return

        with self.caminho.open("w", encoding="utf-8", newline="") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow(self.colunas)


class RegistroMemoria:
    """Adapter em memória; útil para testes e para trocar o CSV sem I/O."""

    def __init__(self, colunas: Sequence[str]) -> None:
        self.colunas = tuple(colunas)
        self.linhas: list[dict[str, Any]] = []

    def adicionar(self, linha: Mapping[str, Any]) -> None:
        _validar_linha(linha, self.colunas)
        self.linhas.append(dict(linha))

    def ler(self) -> list[dict[str, str]]:
        return [
            {
                coluna: "" if linha[coluna] is None else str(linha[coluna])
                for coluna in self.colunas
            }
            for linha in self.linhas
        ]


def _validar_linha(linha: Mapping[str, Any], colunas: Sequence[str]) -> None:
    if not isinstance(linha, Mapping):
        raise TypeError("A linha deve ser um mapeamento com o nome das colunas.")

    faltando = [coluna for coluna in colunas if coluna not in linha]
    sobrando = [coluna for coluna in linha if coluna not in colunas]
    if faltando or sobrando:
        detalhes = []
        if faltando:
            detalhes.append(f"faltando: {', '.join(faltando)}")
        if sobrando:
            detalhes.append(f"sobrando: {', '.join(sobrando)}")
        raise ValueError(
            "Linha incompatível com as colunas do registro "
            f"({'; '.join(detalhes)})."
        )


def ler_csv(caminho: str | Path) -> list[dict[str, str]]:
    """Lê um CSV de registro e devolve uma lista de dicionários."""
    arquivo = Path(caminho)
    if not arquivo.exists():
        return []

    with arquivo.open("r", encoding="utf-8", newline="") as aberto:
        return list(csv.DictReader(aberto))
