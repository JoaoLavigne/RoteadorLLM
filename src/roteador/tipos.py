"""Tipos, enums e contratos compartilhados pelo roteador."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Papel(str, Enum):
    BARATO = "barato"
    FORTE = "forte"


class Faixa(str, Enum):
    CURTA = "curta"
    MEDIA = "media"
    LONGA = "longa"


class MotivoParada(str, Enum):
    COMPLETO = "completo"
    TRUNCADO = "truncado"
    VAZIO = "vazio"


class TipoOcorrencia(str, Enum):
    NENHUMA = "nenhuma"
    VAZIO = "vazio"
    TRUNCADO = "truncado"
    POSSIVELMENTE_TRUNCADO = "possivelmente_truncado"
    LIMITE_CONSUMIDO_PELO_RACIOCINIO = "limite_consumido_pelo_raciocinio"


@dataclass(frozen=True)
class RespostaModelo:
    texto: str
    tokens_entrada: int
    tokens_saida: int
    tokens_raciocinio: int | None
    motivo_parada: MotivoParada
    tempo_ms: float
    modelo: str


@dataclass(frozen=True)
class Decisao:
    papel: Papel
    motivo: str
    faixa_prevista: Faixa | None = None
    limite_tokens: int | None = None


@dataclass(frozen=True)
class Ocorrencia:
    tipo: TipoOcorrencia
    alerta: bool
    motivo: str


@dataclass(frozen=True)
class Resultado:
    resposta: RespostaModelo
    decisao: Decisao
    ocorrencia: Ocorrencia
    custo_estimado_usd: float
    tentativas: int


class Tokenizador(Protocol):
    def contar(self, texto: str) -> int: ...


class Adapter(Protocol):
    nome: str

    def gerar(self, prompt: str, limite_tokens: int | None) -> RespostaModelo: ...


class Roteador(Protocol):
    def decidir(self, pergunta: str, contexto: str = "") -> Decisao: ...
