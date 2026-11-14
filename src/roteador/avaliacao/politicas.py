"""Políticas de roteamento avaliadas offline sobre o dataset."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from roteador.config import Config
from roteador.dados.carregador import LinhaDataset, Nota
from roteador.dados.rotulos import barato_basta
from roteador.tokenizador import TokenizadorAproximado
from roteador.tipos import Papel, Tokenizador

_PRECO_POR_MILHAO = 1_000_000


@dataclass(frozen=True)
class Escolha:
    """Modelo escolhido para uma linha e o resultado observado nele."""

    papel: Papel
    nota: Nota
    tokens_resposta: int


@dataclass(frozen=True)
class ResultadoPolitica:
    """Métricas de uma política sobre o dataset."""

    nome: str
    qualidade_media: float
    custo_estimado_usd: float
    custo_medio_usd: float
    qualidade_relativa_pct: float
    custo_relativo_pct: float
    economia_pct: float
    linhas: int
    aviso: str | None = None

    @property
    def disponivel(self) -> bool:
        return self.aviso is None


def escolha_barato(linha: LinhaDataset) -> Escolha:
    """Escolhe o modelo barato e usa a nota e os tokens dele."""
    return Escolha(
        papel=Papel.BARATO,
        nota=linha.nota_barato,
        tokens_resposta=linha.tokens_resposta_barato,
    )


def escolha_forte(linha: LinhaDataset) -> Escolha:
    """Escolhe o modelo forte e usa a nota e os tokens dele."""
    return Escolha(
        papel=Papel.FORTE,
        nota=linha.nota_forte,
        tokens_resposta=linha.tokens_resposta_forte,
    )


def escolha_perfeito(linha: LinhaDataset) -> Escolha:
    """Escolhe o barato quando ele basta; caso contrário, escolhe o forte."""
    if barato_basta(linha):
        return escolha_barato(linha)
    return escolha_forte(linha)


def escolha_forte_limitado(linha: LinhaDataset) -> Escolha:
    """Escolhe o forte com o resultado limitado da linha."""
    if linha.nota_forte_limitado is None or linha.tokens_resposta_forte_limitado is None:
        raise ValueError(
            "A linha não tem nota_forte_limitado e tokens_resposta_forte_limitado."
        )
    return Escolha(
        papel=Papel.FORTE,
        nota=linha.nota_forte_limitado,
        tokens_resposta=linha.tokens_resposta_forte_limitado,
    )


def avaliar_politicas(
    linhas: Sequence[LinhaDataset],
    cfg: Config,
    tokenizador: Tokenizador | None = None,
    proporcao: float = 0.5,
    semente: int = 42,
    extras: Mapping[str, Callable[[LinhaDataset], Escolha]] | None = None,
) -> list[ResultadoPolitica]:
    """Compara as políticas padrão e devolve as métricas de cada uma.

    A proporção do aleatório é a fatia de linhas enviada ao modelo barato.
    As políticas extras recebem uma linha e devolvem a escolha correspondente.
    """
    lista = list(linhas)
    if not lista:
        raise ValueError("Não é possível avaliar políticas sem linhas no dataset.")

    if not 0 <= proporcao <= 1:
        raise ValueError("A proporção do aleatório deve estar entre 0 e 1.")

    tokenizador = tokenizador or TokenizadorAproximado(cfg.divisor_aproximado)

    escolhas_fortes = [escolha_forte(linha) for linha in lista]
    referencia = _medir(
        "sempre_forte", lista, escolhas_fortes, cfg, tokenizador, None
    )

    resultados = [
        _medir(
            "sempre_barato",
            lista,
            [escolha_barato(linha) for linha in lista],
            cfg,
            tokenizador,
            referencia,
        ),
        referencia,
        _medir(
            "aleatorio",
            lista,
            _escolhas_aleatorias(lista, proporcao, semente),
            cfg,
            tokenizador,
            referencia,
        ),
        _medir(
            "perfeito",
            lista,
            [escolha_perfeito(linha) for linha in lista],
            cfg,
            tokenizador,
            referencia,
        ),
        _medir(
            "v1",
            lista,
            [_escolha_v1(linha, tokenizador, cfg) for linha in lista],
            cfg,
            tokenizador,
            referencia,
        ),
    ]

    faltando_limitado = [
        linha
        for linha in lista
        if linha.nota_forte_limitado is None
        or linha.tokens_resposta_forte_limitado is None
    ]
    if faltando_limitado:
        resultados.append(
            ResultadoPolitica(
                nome="forte_limitado",
                qualidade_media=0.0,
                custo_estimado_usd=0.0,
                custo_medio_usd=0.0,
                qualidade_relativa_pct=0.0,
                custo_relativo_pct=0.0,
                economia_pct=0.0,
                linhas=len(lista),
                aviso=(
                    "Política forte_limitado ignorada: "
                    f"{len(faltando_limitado)} linha(s) sem "
                    "nota_forte_limitado ou tokens_resposta_forte_limitado."
                ),
            )
        )
    else:
        resultados.append(
            _medir(
                "forte_limitado",
                lista,
                [escolha_forte_limitado(linha) for linha in lista],
                cfg,
                tokenizador,
                referencia,
            )
        )

    for nome, escolher in (extras or {}).items():
        resultados.append(
            _medir(
                nome,
                lista,
                [escolher(linha) for linha in lista],
                cfg,
                tokenizador,
                referencia,
            )
        )

    return resultados


def _medir(
    nome: str,
    linhas: Sequence[LinhaDataset],
    escolhas: Sequence[Escolha],
    cfg: Config,
    tokenizador: Tokenizador,
    referencia: ResultadoPolitica | None,
) -> ResultadoPolitica:
    qualidade_media = sum(int(escolha.nota) for escolha in escolhas) / (
        2 * len(linhas)
    )
    custo_total = sum(
        _custo(escolha, _tokens_entrada(linha, tokenizador), cfg)
        for linha, escolha in zip(linhas, escolhas)
    )

    if referencia is None:
        qualidade_relativa = 100.0
        custo_relativo = 100.0
        economia = 0.0
    else:
        qualidade_relativa = _percentual(
            qualidade_media, referencia.qualidade_media
        )
        custo_relativo = _percentual(
            custo_total, referencia.custo_estimado_usd
        )
        economia = 100.0 - custo_relativo

    return ResultadoPolitica(
        nome=nome,
        qualidade_media=qualidade_media,
        custo_estimado_usd=custo_total,
        custo_medio_usd=custo_total / len(linhas),
        qualidade_relativa_pct=qualidade_relativa,
        custo_relativo_pct=custo_relativo,
        economia_pct=economia,
        linhas=len(linhas),
    )


def _tokens_entrada(linha: LinhaDataset, tokenizador: Tokenizador) -> int:
    return tokenizador.contar(linha.pergunta) + (linha.contexto_tokens or 0)


def _custo(escolha: Escolha, tokens_entrada: int, cfg: Config) -> float:
    preco_entrada, preco_saida = cfg.preco_ref[escolha.papel.value]
    return (
        tokens_entrada * preco_entrada
        + escolha.tokens_resposta * preco_saida
    ) / _PRECO_POR_MILHAO


def _escolhas_aleatorias(
    linhas: Sequence[LinhaDataset], proporcao: float, semente: int
) -> list[Escolha]:
    embaralhador = random.Random(semente)
    return [
        escolha_barato(linha)
        if embaralhador.random() < proporcao
        else escolha_forte(linha)
        for linha in linhas
    ]


def _escolha_v1(
    linha: LinhaDataset, tokenizador: Tokenizador, cfg: Config
) -> Escolha:
    tokens_pergunta = tokenizador.contar(linha.pergunta)
    tokens_contexto = linha.contexto_tokens or 0

    if tokens_pergunta + tokens_contexto > cfg.janela_contexto_barato:
        return escolha_forte(linha)
    if tokens_pergunta > cfg.limiar_tokens_pergunta:
        return escolha_forte(linha)
    return escolha_barato(linha)


def _percentual(valor: float, base: float) -> float:
    if base == 0:
        return 0.0
    return 100.0 * valor / base
