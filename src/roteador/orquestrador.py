"""Orquestrador: junta roteador, adapters, detector e armazenamento.

O armazenamento entra pela interface `Registro`, não pelo CSV. Hoje o
adapter usado é `RegistroCsv`; para migrar para banco de dados, basta
criar outro adapter com `adicionar`/`ler` e passar no lugar. A lógica de
roteamento, retry, custo e classificação não muda.

Retry e fallback de comunicação ficam em `comunicacao.py`. O orquestrador
só consome o `ResultadoComunicacao`, que já informa o papel e o limite que
foram usados de verdade.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from math import ceil

from roteador.comunicacao import ExecutorComunicacao, ResultadoComunicacao
from roteador.config import Config
from roteador.detector import classificar
from roteador.limite import montar_prompt, teto_rigido
from roteador.registro import Registro
from roteador.tipos import (
    Adapter,
    Decisao,
    Ocorrencia,
    Papel,
    Resultado,
    RespostaModelo,
    Roteador,
    TipoOcorrencia,
    Tokenizador,
)

_OCORRENCIAS_COM_NOVA_TENTATIVA = {
    TipoOcorrencia.VAZIO,
    TipoOcorrencia.TRUNCADO,
    TipoOcorrencia.LIMITE_CONSUMIDO_PELO_RACIOCINIO,
}

_ACAO_NENHUMA = "nenhuma"
_ACAO_NOVA_TENTATIVA = "nova_tentativa_teto_maior"
_ACAO_ESCALAR = "escalou_ao_forte"

_RESULTADO_OK = "ok"
_RESULTADO_COM_OCORRENCIA = "ainda_com_ocorrencia"


class Orquestrador:
    """Executa uma requisição completa, com no máximo uma nova tentativa."""

    def __init__(
        self,
        roteador: Roteador,
        adapters: Mapping[Papel, Adapter],
        tokenizador: Tokenizador,
        registro: Registro,
        registro_ml: Registro,
        cfg: Config,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.roteador = roteador
        self.adapters = adapters
        self.tokenizador = tokenizador
        self.registro = registro
        self.registro_ml = registro_ml
        self.cfg = cfg
        self.comunicacao = ExecutorComunicacao(
            adapters,
            tentativas=cfg.tentativas_comunicacao,
            espera=cfg.espera_entre_tentativas_s,
            sleep=sleep or time.sleep,
            fallback=cfg.fallback_comunicacao,
        )

    def responder(self, pergunta: str, contexto: str = "") -> Resultado:
        id_requisicao = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        tokens_pergunta = self.tokenizador.contar(pergunta)

        decisao = self.roteador.decidir(pergunta, contexto)
        prompt_base = _montar_prompt_base(pergunta, contexto)
        prompt, teto = self._prompt_e_teto(decisao, prompt_base)

        resultado_1 = self._executar(
            decisao.papel, prompt_base, prompt, teto
        )
        resposta_1 = resultado_1.resposta
        ocorrencia_1 = classificar(
            resposta_1, resultado_1.limite_aplicado, self.cfg
        )
        respostas = [resposta_1]
        papeis = [resultado_1.papel]
        tentativas = resultado_1.tentativas

        if ocorrencia_1.tipo in _OCORRENCIAS_COM_NOVA_TENTATIVA:
            if resultado_1.papel == Papel.BARATO:
                acao = _ACAO_ESCALAR
                papel_retry = Papel.FORTE
                prompt_retry = prompt_base
                teto_retry = None
            else:
                acao = _ACAO_NOVA_TENTATIVA
                papel_retry = Papel.FORTE
                prompt_retry = resultado_1.prompt
                teto_retry = (
                    ceil(
                        resultado_1.limite_aplicado
                        * self.cfg.fator_nova_tentativa
                    )
                    if resultado_1.limite_aplicado is not None
                    else None
                )

            resultado_2 = self._executar(
                papel_retry, prompt_base, prompt_retry, teto_retry
            )
            resposta_2 = resultado_2.resposta
            ocorrencia_2 = classificar(
                resposta_2, resultado_2.limite_aplicado, self.cfg
            )
            respostas.append(resposta_2)
            papeis.append(resultado_2.papel)
            tentativas += resultado_2.tentativas

            resultado_final_1 = (
                _RESULTADO_OK
                if ocorrencia_2.tipo == TipoOcorrencia.NENHUMA
                else _RESULTADO_COM_OCORRENCIA
            )
            self._gravar_ml(
                id_requisicao,
                timestamp,
                decisao,
                resposta_1,
                ocorrencia_1,
                resultado_1.limite_aplicado,
                acao,
                resultado_final_1,
            )

            if ocorrencia_2.tipo != TipoOcorrencia.NENHUMA:
                self._gravar_ml(
                    id_requisicao,
                    timestamp,
                    decisao,
                    resposta_2,
                    ocorrencia_2,
                    resultado_2.limite_aplicado,
                    _ACAO_NENHUMA,
                    _RESULTADO_COM_OCORRENCIA,
                )

            resposta_final = resposta_2
            ocorrencia_final = ocorrencia_2
        else:
            if ocorrencia_1.tipo != TipoOcorrencia.NENHUMA:
                self._gravar_ml(
                    id_requisicao,
                    timestamp,
                    decisao,
                    resposta_1,
                    ocorrencia_1,
                    resultado_1.limite_aplicado,
                    _ACAO_NENHUMA,
                    _RESULTADO_COM_OCORRENCIA,
                )

            resposta_final = resposta_1
            ocorrencia_final = ocorrencia_1

        custo = sum(
            _calcular_custo(resposta, papel, self.cfg)
            for resposta, papel in zip(respostas, papeis)
        )

        self.registro.adicionar(
            {
                "id_requisicao": id_requisicao,
                "timestamp": timestamp,
                "papel": decisao.papel.value,
                "modelo": resposta_final.modelo,
                "tokens_pergunta": tokens_pergunta,
                "tokenizador_usado": _nome_do_tokenizador(self.tokenizador),
                "tokens_entrada": sum(r.tokens_entrada for r in respostas),
                "tokens_saida": sum(r.tokens_saida for r in respostas),
                "tokens_raciocinio": _somar_raciocinio(respostas),
                "tempo_ms": sum(r.tempo_ms for r in respostas),
                "custo_estimado_usd": custo,
                "motivo_decisao": decisao.motivo,
                "faixa_prevista": (
                    decisao.faixa_prevista.value if decisao.faixa_prevista else ""
                ),
                "limite_tokens": (
                    decisao.limite_tokens if decisao.limite_tokens is not None else ""
                ),
                "motivo_parada": resposta_final.motivo_parada.value,
                "ocorrencia": ocorrencia_final.tipo.value,
                "tentativas": tentativas,
            }
        )

        return Resultado(
            resposta=resposta_final,
            decisao=decisao,
            ocorrencia=ocorrencia_final,
            custo_estimado_usd=custo,
            tentativas=tentativas,
        )

    def _prompt_e_teto(
        self, decisao: Decisao, prompt_base: str
    ) -> tuple[str, int | None]:
        if decisao.papel != Papel.FORTE or decisao.limite_tokens is None:
            return prompt_base, None

        x = decisao.limite_tokens
        return montar_prompt(prompt_base, x), teto_rigido(x, self.cfg)

    def _executar(
        self,
        papel: Papel,
        prompt_base: str,
        prompt_limitado: str,
        teto: int | None,
    ) -> ResultadoComunicacao:
        return self.comunicacao.executar(
            papel, prompt_base, prompt_limitado, teto
        )

    def _gravar_ml(
        self,
        id_requisicao: str,
        timestamp: str,
        decisao: Decisao,
        resposta: RespostaModelo,
        ocorrencia: Ocorrencia,
        limite_aplicado: int | None,
        acao: str,
        resultado_final: str,
    ) -> None:
        self.registro_ml.adicionar(
            {
                "id_requisicao": id_requisicao,
                "timestamp": timestamp,
                "modelo": resposta.modelo,
                "faixa_prevista": (
                    decisao.faixa_prevista.value if decisao.faixa_prevista else ""
                ),
                "limite_aplicado": (
                    limite_aplicado if limite_aplicado is not None else ""
                ),
                "tokens_saida_reais": resposta.tokens_saida,
                "tokens_raciocinio": (
                    resposta.tokens_raciocinio
                    if resposta.tokens_raciocinio is not None
                    else ""
                ),
                "motivo_parada": resposta.motivo_parada.value,
                "tipo_ocorrencia": ocorrencia.tipo.value,
                "acao": acao,
                "resultado_final": resultado_final,
            }
        )


def _montar_prompt_base(pergunta: str, contexto: str) -> str:
    if contexto:
        return f"{contexto}\n\n{pergunta}"
    return pergunta


def _nome_do_tokenizador(tokenizador: Tokenizador) -> str:
    nome = getattr(tokenizador, "tokenizador_usado", None)
    if nome is None:
        nome = getattr(tokenizador, "nome", "desconhecido")
    return nome


def _somar_raciocinio(respostas: list[RespostaModelo]) -> int | None:
    valores = [
        resposta.tokens_raciocinio
        for resposta in respostas
        if resposta.tokens_raciocinio is not None
    ]
    return sum(valores) if valores else None


def _calcular_custo(
    resposta: RespostaModelo, papel: Papel, cfg: Config
) -> float:
    preco_entrada, preco_saida = cfg.preco_ref[papel.value]
    tokens_saida = resposta.tokens_saida + (resposta.tokens_raciocinio or 0)
    return (
        resposta.tokens_entrada * preco_entrada
        + tokens_saida * preco_saida
    ) / 1_000_000
