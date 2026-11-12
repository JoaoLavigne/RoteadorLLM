"""Política de retry e fallback para chamadas aos adapters de LLM."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from roteador.adapters.base import ErroAdapter
from roteador.tipos import Adapter, Papel, RespostaModelo

__all__ = ["ExecutorComunicacao", "ResultadoComunicacao"]


@dataclass(frozen=True)
class ResultadoComunicacao:
    """Resposta obtida e o que aconteceu no caminho até ela."""

    resposta: RespostaModelo
    papel: Papel
    prompt: str
    limite_aplicado: int | None
    tentativas: int
    fallback: bool
    erros: tuple[str, ...] = ()


class ExecutorComunicacao:
    """Tenta de novo erros transitórios e troca de papel quando necessário."""

    def __init__(
        self,
        adapters: Mapping[Papel, Adapter],
        tentativas: int = 3,
        espera: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        fallback: bool = True,
    ) -> None:
        if tentativas < 1:
            raise ValueError("tentativas deve ser pelo menos 1.")
        if espera < 0:
            raise ValueError("espera não pode ser negativa.")

        self.adapters = adapters
        self.tentativas = tentativas
        self.espera = espera
        self._sleep = sleep
        self.fallback = fallback

    def executar(
        self,
        papel: Papel,
        prompt_base: str,
        prompt_limitado: str,
        teto: int | None,
    ) -> ResultadoComunicacao:
        prompt, limite = _prompt_e_limite(
            papel, prompt_base, prompt_limitado, teto
        )
        resposta, chamadas, erros = self._tentar(papel, prompt, limite)

        if resposta is not None:
            return ResultadoComunicacao(
                resposta=resposta,
                papel=papel,
                prompt=prompt,
                limite_aplicado=limite,
                tentativas=chamadas,
                fallback=False,
                erros=tuple(erros),
            )

        if not self.fallback:
            raise ErroAdapter(
                _mensagem_falha(papel, chamadas, erros),
                transitorio=False,
            )

        outro = _outro_papel(papel)
        prompt_outro = prompt_base
        limite_outro = None
        resposta, chamadas_outro, erros_outro = self._tentar(
            outro, prompt_outro, limite_outro
        )

        if resposta is not None:
            return ResultadoComunicacao(
                resposta=resposta,
                papel=outro,
                prompt=prompt_outro,
                limite_aplicado=limite_outro,
                tentativas=chamadas + chamadas_outro,
                fallback=True,
                erros=tuple(erros + erros_outro),
            )

        raise ErroAdapter(
            "Falha de comunicação com os dois papéis: "
            f"{_mensagem_falha(papel, chamadas, erros)}; "
            f"{_mensagem_falha(outro, chamadas_outro, erros_outro)}",
            transitorio=False,
        )

    def _tentar(
        self,
        papel: Papel,
        prompt: str,
        limite: int | None,
    ) -> tuple[RespostaModelo | None, int, list[str]]:
        adapter = self._adapter(papel)
        erros: list[str] = []

        for numero in range(1, self.tentativas + 1):
            try:
                return adapter.gerar(prompt, limite), numero, erros
            except ErroAdapter as exc:
                erros.append(f"{adapter.nome}: {exc}")
                if not exc.transitorio:
                    break
                if numero < self.tentativas:
                    self._sleep(self.espera)

        return None, len(erros), erros

    def _adapter(self, papel: Papel) -> Adapter:
        try:
            return self.adapters[papel]
        except KeyError:
            raise ErroAdapter(
                f"Nenhum adapter configurado para o papel {papel.value!r}.",
                transitorio=False,
            ) from None


def _prompt_e_limite(
    papel: Papel,
    prompt_base: str,
    prompt_limitado: str,
    teto: int | None,
) -> tuple[str, int | None]:
    if papel == Papel.FORTE and teto is not None:
        return prompt_limitado, teto
    return prompt_base, None


def _outro_papel(papel: Papel) -> Papel:
    return Papel.FORTE if papel == Papel.BARATO else Papel.BARATO


def _mensagem_falha(papel: Papel, chamadas: int, erros: list[str]) -> str:
    detalhes = " | ".join(erros) if erros else "sem detalhes"
    return f"{papel.value} falhou após {chamadas} tentativa(s): {detalhes}"
