"""Interface de linha de comando do roteador."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from roteador.adapters.base import Adapter, ErroAdapter
from roteador.adapters.gemini import AdapterGemini
from roteador.adapters.ollama import AdapterOllama
from roteador.avaliacao.curva import gerar_curva
from roteador.avaliacao.politicas import (
    Escolha,
    avaliar_politicas,
    escolha_barato,
    escolha_forte,
)
from roteador.avaliacao.relatorio import (
    gerar_relatorio_alertas,
    tabela_markdown_alertas,
)
from roteador.config import Config, carregar_config
from roteador.dados.carregador import LinhaDataset, carregar_dataset
from roteador.dados.rotulos import (
    barato_basta,
    dividir_por_documento,
    faixa_da_linha,
)
from roteador.ml.classificadores import (
    CLASSES_B,
    ErroModelo,
    ModeloRoteador,
    avaliar as avaliar_classificadores,
    carregar_modelo,
    salvar_modelo,
)
from roteador.ml.embedder import Embedder, EmbedderOllama, ErroEmbedder
from roteador.ml.roteador_v2 import (
    RoteadorV2,
    montar_features_linhas,
    treinar_modelo_v2,
)
from roteador.orquestrador import Orquestrador
from roteador.registro import (
    COLUNAS_ML,
    COLUNAS_REGISTRO,
    RegistroCsv,
    ler_csv,
)
from roteador.roteador_v1 import RoteadorV1
from roteador.tokenizador import ErroTokenizador, criar_tokenizador
from roteador.tipos import (
    Decisao,
    Papel,
    Resultado,
    RespostaModelo,
    Roteador,
    Tokenizador,
)

COLUNAS_COLETA = (
    "id",
    "documento",
    "tipo",
    "pergunta",
    "contexto_tokens",
    "politica",
    "papel",
    "modelo",
    "resposta",
    "tokens_pergunta",
    "tokens_entrada",
    "tokens_saida",
    "tokens_raciocinio",
    "tempo_ms",
    "custo_estimado_usd",
    "motivo_parada",
    "ocorrencia",
    "tentativas",
    "nota",
)

_POLITICAS_COLETA = ("barato", "forte", "forte_limitado", "v1", "v2")


class _AdapterPreguicoso:
    """Cria o adapter real na primeira chamada."""

    def __init__(self, nome: str, fabrica: Callable[[], Adapter]) -> None:
        self.nome = nome
        self._fabrica = fabrica
        self._adapter: Adapter | None = None

    def gerar(
        self, prompt: str, limite_tokens: int | None = None
    ) -> RespostaModelo:
        if self._adapter is None:
            self._adapter = self._fabrica()
        return self._adapter.gerar(prompt, limite_tokens)


class _RoteadorFixo:
    """Roteador de coleta que sempre devolve a mesma decisão."""

    def __init__(
        self,
        papel: Papel,
        motivo: str,
        limite_tokens: int | None = None,
    ) -> None:
        self.papel = papel
        self.motivo = motivo
        self.limite_tokens = limite_tokens

    def decidir(self, pergunta: str, contexto: str = "") -> Decisao:
        return Decisao(
            papel=self.papel,
            motivo=self.motivo,
            limite_tokens=self.limite_tokens,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    cfg: Config | None = None,
    adapters: Mapping[Papel, Adapter] | None = None,
    tokenizador: Tokenizador | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Executa um subcomando e devolve o código de saída."""
    parser = _criar_parser()
    args = parser.parse_args(argv)

    try:
        if cfg is None:
            cfg = carregar_config(args.config)
        if tokenizador is None:
            tokenizador = criar_tokenizador(cfg)

        if args.comando == "rotear":
            return _cmd_rotear(args, cfg, adapters, tokenizador)
        if args.comando == "coletar":
            return _cmd_coletar(args, cfg, adapters, tokenizador, embedder)
        if args.comando == "avaliar":
            return _cmd_avaliar(args, cfg, tokenizador, embedder)
        if args.comando == "treinar":
            return _cmd_treinar(args, cfg, tokenizador, embedder)
    except (
        ErroAdapter,
        ErroEmbedder,
        ErroModelo,
        ErroTokenizador,
        FileNotFoundError,
        ValueError,
        OSError,
    ) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    parser.error("Informe um subcomando.")
    return 2


def _criar_parser() -> argparse.ArgumentParser:
    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument(
        "--config",
        help="Arquivo JSON de configuração que sobrescreve os padrões.",
    )

    parser = argparse.ArgumentParser(
        prog="roteador",
        description="Roteador de requisições entre modelos barato e forte.",
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    rotear = subcomandos.add_parser(
        "rotear", parents=[comum], help="Responde uma pergunta."
    )
    rotear.add_argument("pergunta", help="Pergunta a ser respondida.")
    rotear.add_argument(
        "--contexto",
        help="Arquivo com o contexto da pergunta.",
    )

    coletar = subcomandos.add_parser(
        "coletar",
        parents=[comum],
        help="Coleta respostas para anotação manual.",
    )
    coletar.add_argument(
        "--politica",
        choices=_POLITICAS_COLETA,
        required=True,
        help="Política usada para coletar as respostas.",
    )
    coletar.add_argument("--dataset", help="Dataset de entrada.")
    coletar.add_argument("--saida", help="Arquivo CSV de saída da coleta.")

    avaliar = subcomandos.add_parser(
        "avaliar",
        parents=[comum],
        help="Gera a curva de custo × qualidade e o relatório de alertas.",
    )
    avaliar.add_argument("--dataset", help="Dataset de entrada.")
    avaliar.add_argument("--curva", help="Arquivo CSV de saída da curva.")
    avaliar.add_argument(
        "--modelo",
        help="Modelo v2 usado para incluir a política v2 na curva.",
    )

    treinar = subcomandos.add_parser(
        "treinar",
        parents=[comum],
        help="Treina os classificadores A e B do roteador v2.",
    )
    treinar.add_argument("--dataset", help="Dataset de entrada.")
    treinar.add_argument("--saida", help="Arquivo joblib de saída do modelo.")

    return parser


def _cmd_rotear(
    args: argparse.Namespace,
    cfg: Config,
    adapters: Mapping[Papel, Adapter] | None,
    tokenizador: Tokenizador,
) -> int:
    contexto = ""
    if args.contexto:
        contexto = Path(args.contexto).read_text(encoding="utf-8")

    mapa = adapters or _montar_adapters(cfg, {Papel.BARATO, Papel.FORTE})
    roteador = RoteadorV1(tokenizador, cfg)
    orquestrador = _montar_orquestrador(roteador, mapa, tokenizador, cfg)
    resultado = orquestrador.responder(args.pergunta, contexto)

    print(f"Papel: {resultado.decisao.papel.value}")
    print(f"Motivo: {resultado.decisao.motivo}")
    if resultado.decisao.faixa_prevista is not None:
        print(f"Faixa prevista: {resultado.decisao.faixa_prevista.value}")
    if resultado.decisao.limite_tokens is not None:
        print(f"Limite de tokens: {resultado.decisao.limite_tokens}")
    print(f"Modelo: {resultado.resposta.modelo}")
    print(f"Custo estimado: US$ {resultado.custo_estimado_usd:.6f}")
    print(f"Ocorrência: {resultado.ocorrencia.tipo.value}")
    print(f"Tentativas: {resultado.tentativas}")
    print("Resposta:")
    print(resultado.resposta.texto)
    return 0


def _cmd_coletar(
    args: argparse.Namespace,
    cfg: Config,
    adapters: Mapping[Papel, Adapter] | None,
    tokenizador: Tokenizador,
    embedder: Embedder | None = None,
) -> int:
    dataset = Path(args.dataset or cfg.caminho_dataset)
    linhas = carregar_dataset(dataset)
    roteador = _roteador_de_coleta(
        args.politica, cfg, tokenizador, embedder
    )
    papeis = _papeis_da_politica(args.politica)
    mapa = adapters or _montar_adapters(cfg, papeis)
    orquestrador = _montar_orquestrador(roteador, mapa, tokenizador, cfg)

    saida = Path(args.saida) if args.saida else _saida_coleta_padrao(cfg, args.politica)
    saida.parent.mkdir(parents=True, exist_ok=True)

    with saida.open("w", encoding="utf-8", newline="") as aberto:
        escritor = csv.DictWriter(aberto, fieldnames=COLUNAS_COLETA)
        escritor.writeheader()

        for linha in linhas:
            resultado = orquestrador.responder(linha.pergunta)
            escritor.writerow(_linha_coleta(linha, args.politica, resultado, tokenizador))
            aberto.flush()

    print(f"{len(linhas)} resposta(s) gravada(s) em {saida}.")
    return 0


def _cmd_avaliar(
    args: argparse.Namespace,
    cfg: Config,
    tokenizador: Tokenizador,
    embedder: Embedder | None = None,
) -> int:
    dataset = Path(args.dataset or cfg.caminho_dataset)
    linhas = carregar_dataset(dataset)

    extras: dict[str, Callable[[LinhaDataset], Escolha]] = {}
    caminho_modelo = Path(args.modelo or cfg.caminho_modelo_v2)
    aviso_v2: str | None = None
    if caminho_modelo.exists():
        try:
            embedder = embedder or EmbedderOllama(cfg.embedder_modelo)
            modelo = carregar_modelo(
                caminho_modelo, embedder_esperado=embedder
            )
            extras["v2"] = _avaliador_v2(
                modelo, embedder, tokenizador, cfg
            )
        except (ErroModelo, ErroEmbedder) as exc:
            aviso_v2 = f"Política v2 ignorada: {exc}"
    else:
        aviso_v2 = (
            "Política v2 ignorada: modelo não encontrado em "
            f"{caminho_modelo}. Rode 'roteador treinar' primeiro."
        )

    resultados = avaliar_politicas(
        linhas, cfg, tokenizador=tokenizador, extras=extras
    )

    curva = Path(args.curva) if args.curva else _curva_padrao(cfg)
    tabela = gerar_curva(resultados, curva)
    print(tabela)
    print(f"\nCurva gravada em {curva}.")

    if aviso_v2:
        print(f"Aviso: {aviso_v2}", file=sys.stderr)

    for resultado in resultados:
        if resultado.aviso:
            print(f"Aviso: {resultado.aviso}", file=sys.stderr)

    registro = ler_csv(cfg.caminho_registro)
    ml = ler_csv(cfg.caminho_ml)
    relatorio = gerar_relatorio_alertas(registro, ml, cfg)
    print("\nRelatório de alertas:")
    print(tabela_markdown_alertas(relatorio))
    return 0


def _cmd_treinar(
    args: argparse.Namespace,
    cfg: Config,
    tokenizador: Tokenizador,
    embedder: Embedder | None = None,
) -> int:
    dataset = Path(args.dataset or cfg.caminho_dataset)
    linhas = carregar_dataset(dataset)
    embedder = embedder or EmbedderOllama(cfg.embedder_modelo)

    treino, teste = dividir_por_documento(linhas)
    modelo = treinar_modelo_v2(treino, embedder, tokenizador, cfg)

    saida = Path(args.saida) if args.saida else Path(cfg.caminho_modelo_v2)
    salvar_modelo(modelo, saida)
    print(
        f"Modelo v2 treinado com {len(treino)} linha(s) de treino "
        f"e {len(teste)} de teste. Modelo gravado em {saida}."
    )

    if teste:
        features = montar_features_linhas(teste, embedder, tokenizador)
        metricas_a = avaliar_classificadores(
            modelo.modelo_a,
            features,
            [barato_basta(linha) for linha in teste],
        )
        metricas_b = avaliar_classificadores(
            modelo.modelo_b,
            features,
            [faixa_da_linha(linha, modelo.cortes).value for linha in teste],
            classes=CLASSES_B,
        )
        print(f"Acurácia do classificador A: {metricas_a['acuracia']:.3f}")
        print(f"Acurácia do classificador B: {metricas_b['acuracia']:.3f}")
    return 0


def _avaliador_v2(
    modelo: ModeloRoteador,
    embedder: Embedder,
    tokenizador: Tokenizador,
    cfg: Config,
) -> Callable[[LinhaDataset], Escolha]:
    roteador = RoteadorV2(
        embedder,
        modelo.modelo_a,
        modelo.modelo_b,
        tokenizador,
        cfg,
    )

    def escolher(linha: LinhaDataset) -> Escolha:
        decisao = roteador.decidir(linha.pergunta)
        if decisao.papel == Papel.BARATO:
            return escolha_barato(linha)
        return escolha_forte(linha)

    return escolher


def _roteador_de_coleta(
    politica: str,
    cfg: Config,
    tokenizador: Tokenizador,
    embedder: Embedder | None = None,
) -> Roteador:
    if politica == "barato":
        return _RoteadorFixo(Papel.BARATO, "coleta: sempre barato")
    if politica == "forte":
        return _RoteadorFixo(Papel.FORTE, "coleta: sempre forte")
    if politica == "forte_limitado":
        if cfg.x_tokens_forte_limitado is None:
            raise ValueError(
                "Defina x_tokens_forte_limitado para coletar a política "
                "forte_limitado."
            )
        return _RoteadorFixo(
            Papel.FORTE,
            "coleta: forte limitado",
            limite_tokens=cfg.x_tokens_forte_limitado,
        )
    if politica == "v2":
        caminho = Path(cfg.caminho_modelo_v2)
        if not caminho.exists():
            raise ValueError(
                f"Modelo v2 não encontrado em {caminho}. "
                "Rode 'roteador treinar' primeiro."
            )
        embedder = embedder or EmbedderOllama(cfg.embedder_modelo)
        modelo = carregar_modelo(caminho, embedder_esperado=embedder)
        return RoteadorV2(
            embedder,
            modelo.modelo_a,
            modelo.modelo_b,
            tokenizador,
            cfg,
        )
    return RoteadorV1(tokenizador, cfg)


def _papeis_da_politica(politica: str) -> set[Papel]:
    if politica == "barato":
        return {Papel.BARATO}
    if politica in {"forte", "forte_limitado"}:
        return {Papel.FORTE}
    return {Papel.BARATO, Papel.FORTE}


def _montar_adapters(
    cfg: Config, papeis: set[Papel]
) -> dict[Papel, Adapter]:
    fabricas: dict[Papel, Callable[[], Adapter]] = {
        Papel.BARATO: lambda: AdapterOllama(modelo=cfg.modelo_barato),
        Papel.FORTE: AdapterGemini,
    }
    nomes = {Papel.BARATO: cfg.modelo_barato, Papel.FORTE: "gemini"}
    return {
        papel: _AdapterPreguicoso(nomes[papel], fabricas[papel])
        for papel in papeis
    }


def _montar_orquestrador(
    roteador: Roteador,
    adapters: Mapping[Papel, Adapter],
    tokenizador: Tokenizador,
    cfg: Config,
) -> Orquestrador:
    registro = RegistroCsv(cfg.caminho_registro, COLUNAS_REGISTRO)
    registro_ml = RegistroCsv(cfg.caminho_ml, COLUNAS_ML)
    return Orquestrador(
        roteador,
        adapters,
        tokenizador,
        registro,
        registro_ml,
        cfg,
    )


def _linha_coleta(
    linha: LinhaDataset,
    politica: str,
    resultado: Resultado,
    tokenizador: Tokenizador,
) -> dict[str, object]:
    resposta = resultado.resposta
    return {
        "id": linha.id,
        "documento": linha.documento,
        "tipo": linha.tipo,
        "pergunta": linha.pergunta,
        "contexto_tokens": linha.contexto_tokens if linha.contexto_tokens is not None else "",
        "politica": politica,
        "papel": resultado.decisao.papel.value,
        "modelo": resposta.modelo,
        "resposta": resposta.texto,
        "tokens_pergunta": tokenizador.contar(linha.pergunta),
        "tokens_entrada": resposta.tokens_entrada,
        "tokens_saida": resposta.tokens_saida,
        "tokens_raciocinio": (
            resposta.tokens_raciocinio
            if resposta.tokens_raciocinio is not None
            else ""
        ),
        "tempo_ms": resposta.tempo_ms,
        "custo_estimado_usd": resultado.custo_estimado_usd,
        "motivo_parada": resposta.motivo_parada.value,
        "ocorrencia": resultado.ocorrencia.tipo.value,
        "tentativas": resultado.tentativas,
        "nota": "",
    }


def _saida_coleta_padrao(cfg: Config, politica: str) -> Path:
    return Path(cfg.caminho_registro).parent / f"coleta_{politica}.csv"


def _curva_padrao(cfg: Config) -> Path:
    return Path(cfg.caminho_registro).parent / "curva.csv"


if __name__ == "__main__":
    raise SystemExit(main())
