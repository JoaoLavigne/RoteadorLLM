"""Calibração offline dos limiares dos roteadores v1 e v2.

O ``limiar_tokens_pergunta`` do v1 é calibrado com dados rotulados próprios.
O ``limiar_probabilidade`` do v2 é calibrado com um recorte do RouterBench e
confirmado nos dados próprios. Os dois valores dependem do tokenizador.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from roteador.avaliacao.politicas import (
    ResultadoPolitica,
    avaliar_politicas,
    escolha_barato,
    escolha_forte,
    medir_escolhas,
)
from roteador.config import Config
from roteador.dados.carregador import LinhaDataset
from roteador.dados.rotulos import dividir_por_documento
from roteador.ml.classificadores import ModeloRoteador
from roteador.ml.embedder import Embedder
from roteador.ml.roteador_v2 import montar_features_linhas, probabilidades_barato
from roteador.tipos import Tokenizador

PERCENTIS_CANDIDATOS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
LIMIARES_FIXOS = (100, 200, 300, 500)
QUALIDADE_MINIMA_PCT = 95.0
FRACAO_TESTE = 0.2
FRACAO_VALIDACAO = 0.2
PASSO_LIMIAR_PROBABILIDADE = 0.01

COLUNAS_CALIBRACAO = (
    "limiar_tokens_pergunta",
    "qualidade_media",
    "custo_estimado_usd",
    "qualidade_relativa_pct",
    "custo_relativo_pct",
    "economia_pct",
    "linhas",
    "escolhido",
)

COLUNAS_CALIBRACAO_PROBABILIDADE = (
    "limiar_probabilidade",
    "qualidade_media",
    "custo_estimado_usd",
    "qualidade_relativa_pct",
    "custo_relativo_pct",
    "economia_pct",
    "linhas",
    "escolhido",
)


@dataclass(frozen=True)
class ResultadoLimiar:
    """Métricas do v1 sobre um conjunto para um valor de limiar."""

    limiar: int
    qualidade_media: float
    custo_estimado_usd: float
    qualidade_relativa_pct: float
    custo_relativo_pct: float
    economia_pct: float
    linhas: int


@dataclass(frozen=True)
class CalibracaoLimiar:
    """Resultado completo da calibração, incluindo a validação no teste."""

    limiar_escolhido: int
    qualidade_minima_pct: float
    resultados: tuple[ResultadoLimiar, ...]
    linhas_treino: int
    linhas_validacao: int
    linhas_teste: int
    teste: ResultadoLimiar | None = None
    aviso_selecao: str | None = None
    aviso_teste: str | None = None

    @property
    def aprovado(self) -> bool:
        return (
            self.aviso_selecao is None
            and self.teste is not None
            and self.teste.qualidade_relativa_pct >= self.qualidade_minima_pct
        )


@dataclass(frozen=True)
class ResultadoLimiarProbabilidade:
    """Métricas do v2 sobre um conjunto para um valor de probabilidade."""

    limiar: float
    qualidade_media: float
    custo_estimado_usd: float
    qualidade_relativa_pct: float
    custo_relativo_pct: float
    economia_pct: float
    linhas: int


@dataclass(frozen=True)
class CalibracaoProbabilidade:
    """Resultado da calibração do limiar de probabilidade do v2."""

    limiar_escolhido: float
    qualidade_minima_pct: float
    resultados: tuple[ResultadoLimiarProbabilidade, ...]
    linhas_validacao: int
    linhas_teste: int
    teste: ResultadoLimiarProbabilidade | None = None
    aviso_selecao: str | None = None
    aviso_teste: str | None = None

    @property
    def aprovado(self) -> bool:
        return (
            self.aviso_selecao is None
            and self.teste is not None
            and self.teste.qualidade_relativa_pct >= self.qualidade_minima_pct
        )


def candidatos_limiar(
    linhas: Sequence[LinhaDataset],
    tokenizador: Tokenizador,
    percentis: Sequence[float] = PERCENTIS_CANDIDATOS,
    fixos: Sequence[int] = LIMIARES_FIXOS,
) -> list[int]:
    """Monta os candidatos a partir dos percentis e dos valores fixos."""
    lista = list(linhas)
    if not lista:
        raise ValueError("Não é possível montar candidatos sem linhas no dataset.")

    tokens = sorted(tokenizador.contar(linha.pergunta) for linha in lista)
    candidatos = set()

    for percentil in percentis:
        if not 0 <= percentil <= 1:
            raise ValueError("Os percentis dos candidatos devem estar entre 0 e 1.")
        candidatos.add(max(0, math.ceil(_percentil(tokens, percentil))))

    for fixo in fixos:
        if fixo < 0:
            raise ValueError("Os limiares fixos não podem ser negativos.")
        candidatos.add(int(fixo))

    return sorted(candidatos)


def avaliar_limiar(
    linhas: Sequence[LinhaDataset],
    cfg: Config,
    tokenizador: Tokenizador,
    limiar: int,
) -> ResultadoLimiar:
    """Simula o v1 com um limiar e mede qualidade e custo contra o forte."""
    if limiar < 0:
        raise ValueError("O limiar de tokens não pode ser negativo.")

    lista = list(linhas)
    if not lista:
        raise ValueError("Não é possível avaliar um limiar sem linhas no dataset.")

    cfg_limiar = replace(cfg, limiar_tokens_pergunta=int(limiar))
    resultados = {
        resultado.nome: resultado
        for resultado in avaliar_politicas(
            lista, cfg_limiar, tokenizador=tokenizador
        )
    }
    v1 = resultados["v1"]

    return ResultadoLimiar(
        limiar=int(limiar),
        qualidade_media=v1.qualidade_media,
        custo_estimado_usd=v1.custo_estimado_usd,
        qualidade_relativa_pct=v1.qualidade_relativa_pct,
        custo_relativo_pct=v1.custo_relativo_pct,
        economia_pct=v1.economia_pct,
        linhas=v1.linhas,
    )


def selecionar_limiar(
    resultados: Sequence[ResultadoLimiar],
    qualidade_minima_pct: float = QUALIDADE_MINIMA_PCT,
) -> tuple[ResultadoLimiar, str | None]:
    """Escolhe o maior limiar que mantém a qualidade relativa na meta."""
    lista = list(resultados)
    if not lista:
        raise ValueError("Não é possível escolher um limiar sem candidatos.")
    if not 0 < qualidade_minima_pct <= 100:
        raise ValueError("A qualidade mínima deve estar entre 0 e 100.")

    ordenados = sorted(lista, key=lambda resultado: resultado.limiar)
    viaveis = [
        resultado
        for resultado in ordenados
        if resultado.qualidade_relativa_pct >= qualidade_minima_pct
    ]
    if viaveis:
        return viaveis[-1], None

    melhor = max(
        ordenados,
        key=lambda resultado: (resultado.qualidade_relativa_pct, resultado.limiar),
    )
    aviso = (
        "Nenhum limiar atingiu a qualidade mínima de "
        f"{qualidade_minima_pct:.1f}%. Escolhido o candidato de melhor "
        f"qualidade ({melhor.limiar} tokens, "
        f"{melhor.qualidade_relativa_pct:.1f}%)."
    )
    return melhor, aviso


def dividir_calibracao(
    linhas: Sequence[LinhaDataset],
    fracao_teste: float = FRACAO_TESTE,
    fracao_validacao: float = FRACAO_VALIDACAO,
    semente: int = 42,
) -> tuple[list[LinhaDataset], list[LinhaDataset], list[LinhaDataset]]:
    """Divide treino, validação e teste sem separar linhas do mesmo documento."""
    lista = list(linhas)
    documentos = {linha.documento for linha in lista}
    if len(documentos) < 3:
        raise ValueError(
            "A calibração exige pelo menos 3 documentos para separar treino, "
            f"validação e teste; encontrado(s) {len(documentos)}."
        )

    if not 0 < fracao_teste < 1:
        raise ValueError("A fração de teste deve estar entre 0 e 1.")
    if not 0 < fracao_validacao < 1:
        raise ValueError("A fração de validação deve estar entre 0 e 1.")
    if fracao_teste + fracao_validacao >= 1:
        raise ValueError("As frações de teste e validação devem somar menos de 1.")

    treino_e_validacao, teste = dividir_por_documento(
        lista, fracao_teste=fracao_teste, semente=semente
    )
    fracao_relativa = fracao_validacao / (1 - fracao_teste)
    treino, validacao = dividir_por_documento(
        treino_e_validacao, fracao_teste=fracao_relativa, semente=semente
    )
    return treino, validacao, teste


def calibrar_limiar_tokens(
    linhas: Sequence[LinhaDataset],
    cfg: Config,
    tokenizador: Tokenizador,
    qualidade_minima_pct: float = QUALIDADE_MINIMA_PCT,
    percentis: Sequence[float] = PERCENTIS_CANDIDATOS,
    fixos: Sequence[int] = LIMIARES_FIXOS,
    fracao_teste: float = FRACAO_TESTE,
    fracao_validacao: float = FRACAO_VALIDACAO,
    semente: int = 42,
) -> CalibracaoLimiar:
    """Calibra o limiar na validação e confirma o resultado no teste."""
    treino, validacao, teste = dividir_calibracao(
        linhas,
        fracao_teste=fracao_teste,
        fracao_validacao=fracao_validacao,
        semente=semente,
    )
    if not validacao:
        raise ValueError("A divisão não deixou linhas de validação.")
    if not teste:
        raise ValueError("A divisão não deixou linhas de teste.")

    candidatos = candidatos_limiar(
        validacao, tokenizador, percentis=percentis, fixos=fixos
    )
    resultados = [
        avaliar_limiar(validacao, cfg, tokenizador, candidato)
        for candidato in candidatos
    ]
    escolhido, aviso_selecao = selecionar_limiar(
        resultados, qualidade_minima_pct
    )
    resultado_teste = avaliar_limiar(
        teste, cfg, tokenizador, escolhido.limiar
    )

    aviso_teste = None
    if resultado_teste.qualidade_relativa_pct < qualidade_minima_pct:
        aviso_teste = (
            f"No teste, o limiar {escolhido.limiar} manteve "
            f"{resultado_teste.qualidade_relativa_pct:.1f}% da qualidade do "
            f"forte, abaixo da meta de {qualidade_minima_pct:.1f}%. "
            "Limiar não adotado."
        )

    return CalibracaoLimiar(
        limiar_escolhido=escolhido.limiar,
        qualidade_minima_pct=qualidade_minima_pct,
        resultados=tuple(resultados),
        linhas_treino=len(treino),
        linhas_validacao=len(validacao),
        linhas_teste=len(teste),
        teste=resultado_teste,
        aviso_selecao=aviso_selecao,
        aviso_teste=aviso_teste,
    )


def salvar_calibracao(
    calibracao: CalibracaoLimiar, caminho: str | Path
) -> list[dict[str, object]]:
    """Grava uma linha por candidato no CSV de calibração."""
    arquivo = Path(caminho)
    arquivo.parent.mkdir(parents=True, exist_ok=True)

    linhas = [
        {
            "limiar_tokens_pergunta": resultado.limiar,
            "qualidade_media": resultado.qualidade_media,
            "custo_estimado_usd": resultado.custo_estimado_usd,
            "qualidade_relativa_pct": resultado.qualidade_relativa_pct,
            "custo_relativo_pct": resultado.custo_relativo_pct,
            "economia_pct": resultado.economia_pct,
            "linhas": resultado.linhas,
            "escolhido": int(resultado.limiar == calibracao.limiar_escolhido),
        }
        for resultado in calibracao.resultados
    ]

    with arquivo.open("w", encoding="utf-8", newline="") as aberto:
        escritor = csv.DictWriter(aberto, fieldnames=COLUNAS_CALIBRACAO)
        escritor.writeheader()
        escritor.writerows(linhas)

    return linhas


def salvar_config_limiar(
    calibracao: CalibracaoLimiar, cfg: Config, caminho: str | Path
) -> Path:
    """Grava um JSON aceito por carregar_config com o limiar escolhido."""
    arquivo = Path(caminho)
    arquivo.parent.mkdir(parents=True, exist_ok=True)

    conteudo = {
        "limiar_tokens_pergunta": calibracao.limiar_escolhido,
        "tokenizador": cfg.tokenizador,
    }
    arquivo.write_text(
        json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return arquivo


def tabela_calibracao(calibracao: CalibracaoLimiar) -> str:
    """Monta a tabela em Markdown com todos os candidatos."""
    linhas = [
        "| Limiar | Qualidade média | Qualidade vs. forte | Custo vs. forte | "
        "Economia | Linhas | Escolhido |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for resultado in calibracao.resultados:
        escolhido = "sim" if resultado.limiar == calibracao.limiar_escolhido else ""
        linhas.append(
            "| "
            f"{resultado.limiar} | "
            f"{resultado.qualidade_media:.1%} | "
            f"{resultado.qualidade_relativa_pct:.1f}% | "
            f"{resultado.custo_relativo_pct:.1f}% | "
            f"{resultado.economia_pct:.1f}% | "
            f"{resultado.linhas} | "
            f"{escolhido} |"
        )

    resumo = [
        "",
        f"Linhas: {calibracao.linhas_treino} de treino, "
        f"{calibracao.linhas_validacao} de validação e "
        f"{calibracao.linhas_teste} de teste.",
        f"Limiar escolhido: {calibracao.limiar_escolhido} tokens.",
    ]
    if calibracao.teste is not None:
        resumo.append(
            "Teste: "
            f"{calibracao.teste.qualidade_relativa_pct:.1f}% da qualidade do "
            f"forte, {calibracao.teste.economia_pct:.1f}% de economia, "
            f"{calibracao.teste.linhas} linha(s)."
        )
    return "\n".join([*linhas, *resumo])


def candidatos_limiar_probabilidade(
    passo: float = PASSO_LIMIAR_PROBABILIDADE,
) -> list[float]:
    """Monta uma grade reprodutível de limiares entre 0 e 1."""
    if not 0 < passo <= 1:
        raise ValueError("O passo do limiar deve estar entre 0 e 1.")

    quantidade = round(1 / passo)
    return sorted(
        {round(indice / quantidade, 6) for indice in range(quantidade + 1)}
    )


def avaliar_limiar_probabilidade(
    linhas: Sequence[LinhaDataset],
    probabilidades: Sequence[float],
    cfg: Config,
    tokenizador: Tokenizador,
    limiar: float,
    referencia: ResultadoPolitica | None = None,
) -> ResultadoLimiarProbabilidade:
    """Simula o v2 com um limiar e mede qualidade e custo contra o forte."""
    lista = list(linhas)
    valores = [float(valor) for valor in probabilidades]
    if not lista:
        raise ValueError("Não é possível avaliar um limiar sem linhas.")
    if len(lista) != len(valores):
        raise ValueError(
            "A quantidade de probabilidades difere da quantidade de linhas."
        )
    if not 0 <= limiar <= 1:
        raise ValueError("O limiar de probabilidade deve estar entre 0 e 1.")

    if referencia is None:
        referencia = medir_escolhas(
            "sempre_forte",
            lista,
            [escolha_forte(linha) for linha in lista],
            cfg,
            tokenizador,
        )

    escolhas = [
        escolha_barato(linha) if probabilidade >= limiar else escolha_forte(linha)
        for linha, probabilidade in zip(lista, valores)
    ]
    medido = medir_escolhas(
        "v2", lista, escolhas, cfg, tokenizador, referencia
    )

    return ResultadoLimiarProbabilidade(
        limiar=limiar,
        qualidade_media=medido.qualidade_media,
        custo_estimado_usd=medido.custo_estimado_usd,
        qualidade_relativa_pct=medido.qualidade_relativa_pct,
        custo_relativo_pct=medido.custo_relativo_pct,
        economia_pct=medido.economia_pct,
        linhas=medido.linhas,
    )


def selecionar_limiar_probabilidade(
    resultados: Sequence[ResultadoLimiarProbabilidade],
    qualidade_minima_pct: float = QUALIDADE_MINIMA_PCT,
) -> tuple[ResultadoLimiarProbabilidade, str | None]:
    """Escolhe o menor limiar que mantém a qualidade relativa na meta.

    Em empate de qualidade e custo, escolhe o limiar mais alto, que é a opção
    mais conservadora.
    """
    lista = list(resultados)
    if not lista:
        raise ValueError("Não é possível escolher um limiar sem candidatos.")
    if not 0 < qualidade_minima_pct <= 100:
        raise ValueError("A qualidade mínima deve estar entre 0 e 100.")

    ordenados = sorted(lista, key=lambda resultado: resultado.limiar)
    viaveis = [
        resultado
        for resultado in ordenados
        if resultado.qualidade_relativa_pct >= qualidade_minima_pct
    ]
    if viaveis:
        escolhido = viaveis[0]
        for candidato in viaveis[1:]:
            if (
                candidato.qualidade_media == escolhido.qualidade_media
                and candidato.custo_estimado_usd == escolhido.custo_estimado_usd
            ):
                escolhido = candidato
            else:
                break
        return escolhido, None

    melhor = max(
        ordenados,
        key=lambda resultado: (
            resultado.qualidade_relativa_pct,
            resultado.limiar,
        ),
    )
    aviso = (
        "Nenhum limiar atingiu a qualidade mínima de "
        f"{qualidade_minima_pct:.1f}%. Escolhido o candidato de melhor "
        f"qualidade ({melhor.limiar:.2f}, "
        f"{melhor.qualidade_relativa_pct:.1f}%)."
    )
    return melhor, aviso


def calibrar_limiar_probabilidade(
    validacao: Sequence[LinhaDataset],
    teste: Sequence[LinhaDataset],
    modelo: ModeloRoteador,
    embedder: Embedder,
    tokenizador: Tokenizador,
    cfg: Config,
    qualidade_minima_pct: float = QUALIDADE_MINIMA_PCT,
    limiares: Sequence[float] | None = None,
) -> CalibracaoProbabilidade:
    """Calibra o limiar na validação e confirma o resultado no teste."""
    linhas_validacao = list(validacao)
    linhas_teste = list(teste)
    if not linhas_validacao:
        raise ValueError("A calibração exige linhas de validação.")
    if not linhas_teste:
        raise ValueError("A calibração exige linhas de teste.")
    if not 0 < qualidade_minima_pct <= 100:
        raise ValueError("A qualidade mínima deve estar entre 0 e 100.")

    if limiares is None:
        candidatos = candidatos_limiar_probabilidade()
    else:
        candidatos = sorted({float(valor) for valor in limiares})
    if not candidatos:
        raise ValueError("A calibração exige pelo menos um limiar candidato.")
    for candidato in candidatos:
        if not 0 <= candidato <= 1:
            raise ValueError(
                "Os limiares candidatos devem estar entre 0 e 1."
            )

    probabilidades_validacao = _probabilidades(
        linhas_validacao, modelo, embedder, tokenizador
    )
    referencia_validacao = medir_escolhas(
        "sempre_forte",
        linhas_validacao,
        [escolha_forte(linha) for linha in linhas_validacao],
        cfg,
        tokenizador,
    )
    resultados = [
        avaliar_limiar_probabilidade(
            linhas_validacao,
            probabilidades_validacao,
            cfg,
            tokenizador,
            candidato,
            referencia=referencia_validacao,
        )
        for candidato in candidatos
    ]
    escolhido, aviso_selecao = selecionar_limiar_probabilidade(
        resultados, qualidade_minima_pct
    )

    probabilidades_teste = _probabilidades(
        linhas_teste, modelo, embedder, tokenizador
    )
    referencia_teste = medir_escolhas(
        "sempre_forte",
        linhas_teste,
        [escolha_forte(linha) for linha in linhas_teste],
        cfg,
        tokenizador,
    )
    resultado_teste = avaliar_limiar_probabilidade(
        linhas_teste,
        probabilidades_teste,
        cfg,
        tokenizador,
        escolhido.limiar,
        referencia=referencia_teste,
    )

    aviso_teste = None
    if resultado_teste.qualidade_relativa_pct < qualidade_minima_pct:
        aviso_teste = (
            f"No teste, o limiar {escolhido.limiar:.2f} manteve "
            f"{resultado_teste.qualidade_relativa_pct:.1f}% da qualidade do "
            f"forte, abaixo da meta de {qualidade_minima_pct:.1f}%. "
            "Limiar não adotado."
        )

    return CalibracaoProbabilidade(
        limiar_escolhido=escolhido.limiar,
        qualidade_minima_pct=qualidade_minima_pct,
        resultados=tuple(resultados),
        linhas_validacao=len(linhas_validacao),
        linhas_teste=len(linhas_teste),
        teste=resultado_teste,
        aviso_selecao=aviso_selecao,
        aviso_teste=aviso_teste,
    )


def salvar_calibracao_probabilidade(
    calibracao: CalibracaoProbabilidade, caminho: str | Path
) -> list[dict[str, object]]:
    """Grava uma linha por candidato no CSV de calibração."""
    arquivo = Path(caminho)
    arquivo.parent.mkdir(parents=True, exist_ok=True)

    linhas = [
        {
            "limiar_probabilidade": resultado.limiar,
            "qualidade_media": resultado.qualidade_media,
            "custo_estimado_usd": resultado.custo_estimado_usd,
            "qualidade_relativa_pct": resultado.qualidade_relativa_pct,
            "custo_relativo_pct": resultado.custo_relativo_pct,
            "economia_pct": resultado.economia_pct,
            "linhas": resultado.linhas,
            "escolhido": int(
                abs(resultado.limiar - calibracao.limiar_escolhido) < 1e-12
            ),
        }
        for resultado in calibracao.resultados
    ]

    with arquivo.open("w", encoding="utf-8", newline="") as aberto:
        escritor = csv.DictWriter(
            aberto, fieldnames=COLUNAS_CALIBRACAO_PROBABILIDADE
        )
        escritor.writeheader()
        escritor.writerows(linhas)

    return linhas


def salvar_config_limiar_probabilidade(
    calibracao: CalibracaoProbabilidade, cfg: Config, caminho: str | Path
) -> Path:
    """Grava um JSON aceito por carregar_config com o limiar escolhido."""
    arquivo = Path(caminho)
    arquivo.parent.mkdir(parents=True, exist_ok=True)

    conteudo = {
        "limiar_probabilidade": calibracao.limiar_escolhido,
        "tokenizador": cfg.tokenizador,
    }
    arquivo.write_text(
        json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return arquivo


def tabela_calibracao_probabilidade(calibracao: CalibracaoProbabilidade) -> str:
    """Monta a tabela em Markdown com todos os candidatos."""
    linhas = [
        "| Limiar | Qualidade média | Qualidade vs. forte | Custo vs. forte | "
        "Economia | Linhas | Escolhido |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for resultado in calibracao.resultados:
        escolhido = (
            "sim"
            if abs(resultado.limiar - calibracao.limiar_escolhido) < 1e-12
            else ""
        )
        linhas.append(
            "| "
            f"{resultado.limiar:.2f} | "
            f"{resultado.qualidade_media:.1%} | "
            f"{resultado.qualidade_relativa_pct:.1f}% | "
            f"{resultado.custo_relativo_pct:.1f}% | "
            f"{resultado.economia_pct:.1f}% | "
            f"{resultado.linhas} | "
            f"{escolhido} |"
        )

    resumo = [
        "",
        f"Linhas: {calibracao.linhas_validacao} de validação e "
        f"{calibracao.linhas_teste} de teste.",
        f"Limiar escolhido: {calibracao.limiar_escolhido:.2f}.",
    ]
    if calibracao.teste is not None:
        resumo.append(
            "Teste: "
            f"{calibracao.teste.qualidade_relativa_pct:.1f}% da qualidade do "
            f"forte, {calibracao.teste.economia_pct:.1f}% de economia, "
            f"{calibracao.teste.linhas} linha(s)."
        )
    return "\n".join([*linhas, *resumo])


def _probabilidades(
    linhas: Sequence[LinhaDataset],
    modelo: ModeloRoteador,
    embedder: Embedder,
    tokenizador: Tokenizador,
) -> list[float]:
    features = montar_features_linhas(linhas, embedder, tokenizador)
    return [
        float(valor)
        for valor in probabilidades_barato(modelo.modelo_a, features)
    ]


def _percentil(valores: Sequence[float], percentil: float) -> float:
    posicao = percentil * (len(valores) - 1)
    inferior = math.floor(posicao)
    superior = math.ceil(posicao)
    if inferior == superior:
        return valores[inferior]

    peso = posicao - inferior
    return valores[inferior] * (1 - peso) + valores[superior] * peso
