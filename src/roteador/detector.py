"""Detector de respostas vazias ou truncadas."""

from __future__ import annotations

from roteador.config import Config
from roteador.tipos import MotivoParada, Ocorrencia, RespostaModelo, TipoOcorrencia

_LIMIAR_PROXIMO_DO_TETO = 0.95
_PONTUACAO_FINAL = ".!?…"
_FECHAMENTOS = "\"'”’)]}"


def classificar(
    resposta: RespostaModelo,
    limite_tokens: int | None,
    cfg: Config,
) -> Ocorrencia:
    """Classifica a resposta seguindo a ordem de regras do guia."""
    if not resposta.texto.strip() and resposta.tokens_saida > 0:
        return Ocorrencia(
            tipo=TipoOcorrencia.VAZIO,
            alerta=True,
            motivo="texto vazio com tokens de saída; provável gasto em raciocínio",
        )

    if resposta.tokens_saida == 0:
        return Ocorrencia(
            tipo=TipoOcorrencia.VAZIO,
            alerta=True,
            motivo="nenhum token de saída, sem erro",
        )

    if limite_tokens is not None and limite_tokens > 0:
        raciocinio = resposta.tokens_raciocinio
        if (
            raciocinio is not None
            and raciocinio >= cfg.limiar_raciocinio * limite_tokens
        ):
            return Ocorrencia(
                tipo=TipoOcorrencia.LIMITE_CONSUMIDO_PELO_RACIOCINIO,
                alerta=True,
                motivo=(
                    "tokens de raciocínio consumiram pelo menos "
                    f"{cfg.limiar_raciocinio:.0%} do teto"
                ),
            )

    if resposta.motivo_parada == MotivoParada.TRUNCADO:
        return Ocorrencia(
            tipo=TipoOcorrencia.TRUNCADO,
            alerta=True,
            motivo="motivo de parada indica truncamento",
        )

    if (
        limite_tokens is not None
        and limite_tokens > 0
        and resposta.tokens_saida >= _LIMIAR_PROXIMO_DO_TETO * limite_tokens
        and not _termina_com_pontuacao(resposta.texto)
    ):
        return Ocorrencia(
            tipo=TipoOcorrencia.POSSIVELMENTE_TRUNCADO,
            alerta=True,
            motivo="saída próxima do teto e texto sem pontuação final",
        )

    return Ocorrencia(
        tipo=TipoOcorrencia.NENHUMA,
        alerta=False,
        motivo="resposta completa",
    )


def _termina_com_pontuacao(texto: str) -> bool:
    texto = texto.rstrip()
    while texto and texto[-1] in _FECHAMENTOS:
        texto = texto[:-1].rstrip()
    return bool(texto) and texto[-1] in _PONTUACAO_FINAL
