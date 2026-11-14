"""Leitura e tipagem do dataset usado na avaliação e no treino.

O dataset pode vir de CSV ou de SQLite. As demais camadas recebem sempre a
mesma lista de ``LinhaDataset`` e não sabem qual armazenamento foi usado.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

COLUNAS_OBRIGATORIAS = (
    "id",
    "documento",
    "tipo",
    "pergunta",
    "nota_barato",
    "nota_forte",
    "tokens_resposta_barato",
    "tokens_resposta_forte",
)

COLUNAS_OPCIONAIS = (
    "contexto_tokens",
    "truncada_forte",
    "nota_forte_limitado",
    "tokens_resposta_forte_limitado",
)

COLUNAS_DATASET = COLUNAS_OBRIGATORIAS + COLUNAS_OPCIONAIS

EXTENSOES_SQLITE = (".db", ".sqlite", ".sqlite3")

TIPOS_SQLITE = {
    "id": "TEXT NOT NULL",
    "documento": "TEXT NOT NULL",
    "tipo": "TEXT NOT NULL",
    "pergunta": "TEXT NOT NULL",
    "nota_barato": "TEXT NOT NULL",
    "nota_forte": "TEXT NOT NULL",
    "tokens_resposta_barato": "INTEGER NOT NULL",
    "tokens_resposta_forte": "INTEGER NOT NULL",
    "contexto_tokens": "INTEGER",
    "truncada_forte": "INTEGER",
    "nota_forte_limitado": "TEXT",
    "tokens_resposta_forte_limitado": "INTEGER",
}

_NOME_TABELA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Nota(IntEnum):
    """Nota de qualidade de uma resposta, com o valor usado nos cálculos."""

    ERRADA = 0
    PARCIAL = 1
    CORRETA = 2

    @classmethod
    def de_texto(cls, valor: Any) -> Nota:
        """Converte texto, número ou outra nota para o enum correspondente."""
        if isinstance(valor, cls):
            return valor

        if isinstance(valor, bool):
            raise ValueError(f"Nota inválida: {valor!r}.")

        if isinstance(valor, int):
            try:
                return cls(valor)
            except ValueError:
                raise ValueError(f"Nota inválida: {valor!r}.") from None

        if isinstance(valor, str):
            chave = valor.strip().lower()
            try:
                return cls[chave.upper()]
            except KeyError:
                pass
            try:
                return cls(int(chave))
            except (ValueError, KeyError):
                pass

        raise ValueError(
            f"Nota inválida: {valor!r}. Use 'correta', 'parcial' ou 'errada'."
        )

    @property
    def texto(self) -> str:
        """Devolve a nota no formato gravado no CSV."""
        return self.name.lower()


@dataclass(frozen=True)
class LinhaDataset:
    """Uma linha do dataset já convertida para os tipos usados pelo código."""

    id: str
    documento: str
    tipo: str
    pergunta: str
    nota_barato: Nota
    nota_forte: Nota
    tokens_resposta_barato: int
    tokens_resposta_forte: int
    contexto_tokens: int | None = None
    truncada_forte: bool = False
    nota_forte_limitado: Nota | None = None
    tokens_resposta_forte_limitado: int | None = None


@runtime_checkable
class CarregadorDataset(Protocol):
    """Contrato de qualquer fonte de dados que produza linhas do dataset."""

    def carregar(self) -> list[LinhaDataset]: ...


class CarregadorCsv:
    """Carrega o dataset a partir de um arquivo CSV."""

    def __init__(self, caminho: str | Path) -> None:
        self.caminho = Path(caminho)

    def carregar(self) -> list[LinhaDataset]:
        arquivo = self.caminho
        if not arquivo.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {arquivo}")

        with arquivo.open("r", encoding="utf-8-sig", newline="") as aberto:
            leitor = csv.DictReader(aberto)
            _validar_colunas(
                leitor.fieldnames,
                f"Cabeçalho do dataset inválido em {arquivo}",
            )

            linhas: list[LinhaDataset] = []
            for dados in leitor:
                numero = leitor.line_num
                if _linha_vazia(dados):
                    continue
                try:
                    linhas.append(_converter_linha(dados))
                except ValueError as exc:
                    raise ValueError(
                        f"Linha {numero} de {arquivo}: {exc}"
                    ) from None

        return linhas


class CarregadorSqlite:
    """Carrega o dataset a partir de uma tabela SQLite."""

    def __init__(self, caminho: str | Path, tabela: str = "dataset") -> None:
        _validar_nome_tabela(tabela)
        self.caminho = Path(caminho)
        self.tabela = tabela

    def carregar(self) -> list[LinhaDataset]:
        arquivo = self.caminho
        if not arquivo.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {arquivo}")

        registros = self._ler_registros()

        linhas: list[LinhaDataset] = []
        for numero, registro in enumerate(registros, start=1):
            dados = dict(registro)
            if _linha_vazia(dados):
                continue
            try:
                linhas.append(_converter_linha(dados))
            except ValueError as exc:
                raise ValueError(
                    f"Registro {numero} da tabela {self.tabela!r} "
                    f"em {arquivo}: {exc}"
                ) from None
        return linhas

    def _ler_registros(self) -> list[sqlite3.Row]:
        try:
            conexao = sqlite3.connect(self.caminho)
        except sqlite3.Error as exc:
            raise ValueError(
                f"Não foi possível abrir o dataset SQLite {self.caminho}: {exc}"
            ) from exc

        conexao.row_factory = sqlite3.Row
        try:
            colunas = self._colunas(conexao)
            _validar_colunas(
                colunas,
                f"Tabela {self.tabela!r} do dataset SQLite {self.caminho}",
            )
            try:
                return conexao.execute(
                    f'SELECT * FROM "{self.tabela}"'
                ).fetchall()
            except sqlite3.Error as exc:
                raise ValueError(
                    f"Não foi possível ler a tabela {self.tabela!r} "
                    f"em {self.caminho}: {exc}"
                ) from exc
        finally:
            conexao.close()

    def _colunas(self, conexao: sqlite3.Connection) -> list[str]:
        try:
            informacoes = conexao.execute(
                f'PRAGMA table_info("{self.tabela}")'
            ).fetchall()
        except sqlite3.Error as exc:
            raise ValueError(
                f"Não foi possível ler a tabela {self.tabela!r} "
                f"em {self.caminho}: {exc}"
            ) from exc

        if not informacoes:
            raise ValueError(
                f"Tabela {self.tabela!r} não encontrada em {self.caminho}."
            )
        return [informacao["name"] for informacao in informacoes]


def criar_carregador(
    caminho: str | Path, tabela: str = "dataset"
) -> CarregadorDataset:
    """Escolhe o carregador pela extensão do arquivo."""
    arquivo = Path(caminho)
    if arquivo.suffix.lower() in EXTENSOES_SQLITE:
        return CarregadorSqlite(arquivo, tabela=tabela)
    return CarregadorCsv(arquivo)


def carregar_dataset(
    caminho: str | Path, tabela: str = "dataset"
) -> list[LinhaDataset]:
    """Carrega o dataset de CSV ou SQLite conforme a extensão do arquivo."""
    return criar_carregador(caminho, tabela=tabela).carregar()


ler_dataset = carregar_dataset


def salvar_dataset_sqlite(
    linhas: Sequence[LinhaDataset],
    caminho: str | Path,
    tabela: str = "dataset",
) -> None:
    """Grava as linhas em uma tabela SQLite, substituindo o conteúdo atual."""
    _validar_nome_tabela(tabela)

    arquivo = Path(caminho)
    arquivo.parent.mkdir(parents=True, exist_ok=True)

    definicoes = ", ".join(
        f'"{coluna}" {TIPOS_SQLITE[coluna]}' for coluna in COLUNAS_DATASET
    )
    criar_tabela = f'CREATE TABLE IF NOT EXISTS "{tabela}" ({definicoes})'
    colunas_sql = ", ".join(f'"{coluna}"' for coluna in COLUNAS_DATASET)
    marcadores = ", ".join("?" for _ in COLUNAS_DATASET)
    inserir = (
        f'INSERT INTO "{tabela}" ({colunas_sql}) VALUES ({marcadores})'
    )
    valores = [_para_tupla(linha) for linha in linhas]

    try:
        conexao = sqlite3.connect(arquivo)
    except sqlite3.Error as exc:
        raise ValueError(
            f"Não foi possível abrir o dataset SQLite {arquivo}: {exc}"
        ) from exc

    try:
        with conexao:
            conexao.execute(criar_tabela)
            conexao.execute(f'DELETE FROM "{tabela}"')
            conexao.executemany(inserir, valores)
    except sqlite3.Error as exc:
        raise ValueError(
            f"Não foi possível gravar a tabela {tabela!r} "
            f"em {arquivo}: {exc}"
        ) from exc
    finally:
        conexao.close()


def _para_tupla(linha: LinhaDataset) -> tuple[Any, ...]:
    nota_limitado = (
        Nota.de_texto(linha.nota_forte_limitado).texto
        if linha.nota_forte_limitado is not None
        else None
    )
    return (
        linha.id,
        linha.documento,
        linha.tipo,
        linha.pergunta,
        Nota.de_texto(linha.nota_barato).texto,
        Nota.de_texto(linha.nota_forte).texto,
        linha.tokens_resposta_barato,
        linha.tokens_resposta_forte,
        linha.contexto_tokens,
        int(linha.truncada_forte),
        nota_limitado,
        linha.tokens_resposta_forte_limitado,
    )


def _validar_nome_tabela(tabela: str) -> None:
    if not isinstance(tabela, str) or not _NOME_TABELA.fullmatch(tabela):
        raise ValueError(
            f"Nome de tabela inválido: {tabela!r}. "
            "Use letras, números e sublinhado, começando por letra ou sublinhado."
        )


def _validar_colunas(campos: Sequence[str] | None, contexto: str) -> None:
    presentes = set(campos or ())
    faltando = [
        coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in presentes
    ]
    if faltando:
        raise ValueError(f"{contexto}: faltando {', '.join(faltando)}.")


def _linha_vazia(dados: Mapping[str, Any]) -> bool:
    return all(valor is None or str(valor).strip() == "" for valor in dados.values())


def _converter_linha(dados: Mapping[str, Any]) -> LinhaDataset:
    return LinhaDataset(
        id=_texto(dados, "id"),
        documento=_texto(dados, "documento"),
        tipo=_texto(dados, "tipo"),
        pergunta=_texto(dados, "pergunta"),
        nota_barato=_nota(dados, "nota_barato", obrigatoria=True),
        nota_forte=_nota(dados, "nota_forte", obrigatoria=True),
        tokens_resposta_barato=_inteiro(dados, "tokens_resposta_barato"),
        tokens_resposta_forte=_inteiro(dados, "tokens_resposta_forte"),
        contexto_tokens=_inteiro(dados, "contexto_tokens", opcional=True),
        truncada_forte=_booleano(dados, "truncada_forte"),
        nota_forte_limitado=_nota(dados, "nota_forte_limitado", obrigatoria=False),
        tokens_resposta_forte_limitado=_inteiro(
            dados, "tokens_resposta_forte_limitado", opcional=True
        ),
    )


def _texto(dados: Mapping[str, Any], campo: str) -> str:
    valor = dados.get(campo)
    if valor is None:
        return ""
    return str(valor).strip()


def _nota(
    dados: Mapping[str, Any], campo: str, obrigatoria: bool
) -> Nota | None:
    valor = dados.get(campo)
    if valor is None or str(valor).strip() == "":
        if obrigatoria:
            raise ValueError(f"O campo {campo} é obrigatório.")
        return None

    try:
        return Nota.de_texto(valor)
    except ValueError as exc:
        raise ValueError(f"O campo {campo} tem {exc}") from None


def _inteiro(
    dados: Mapping[str, Any], campo: str, opcional: bool = False
) -> int | None:
    valor = dados.get(campo)
    if valor is None or str(valor).strip() == "":
        if opcional:
            return None
        raise ValueError(f"O campo {campo} é obrigatório.")

    texto = str(valor).strip()
    try:
        numero = int(texto)
    except ValueError:
        raise ValueError(
            f"O campo {campo} deve ser um inteiro não negativo; recebido {texto!r}."
        ) from None

    if numero < 0:
        raise ValueError(
            f"O campo {campo} deve ser um inteiro não negativo; recebido {texto!r}."
        )
    return numero


def _booleano(dados: Mapping[str, Any], campo: str) -> bool:
    valor = dados.get(campo)
    if valor is None or str(valor).strip() == "":
        return False

    texto = str(valor).strip().lower()
    if texto in {"true", "1", "sim", "s"}:
        return True
    if texto in {"false", "0", "nao", "não", "n"}:
        return False
    raise ValueError(
        f"O campo {campo} deve ser verdadeiro ou falso; recebido {texto!r}."
    )
