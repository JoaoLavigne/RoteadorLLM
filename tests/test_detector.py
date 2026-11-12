from roteador.config import Config
from roteador.detector import classificar
from roteador.tipos import MotivoParada, RespostaModelo, TipoOcorrencia


def _resposta(
    texto: str = "resposta.",
    tokens_saida: int = 10,
    tokens_raciocinio: int | None = None,
    motivo_parada: MotivoParada = MotivoParada.COMPLETO,
) -> RespostaModelo:
    return RespostaModelo(
        texto=texto,
        tokens_entrada=1,
        tokens_saida=tokens_saida,
        tokens_raciocinio=tokens_raciocinio,
        motivo_parada=motivo_parada,
        tempo_ms=1.0,
        modelo="teste",
    )


def test_texto_vazio_com_tokens_de_saida_e_vazio():
    ocorrencia = classificar(_resposta(texto="", tokens_saida=5), 100, Config())

    assert ocorrencia.tipo == TipoOcorrencia.VAZIO
    assert ocorrencia.alerta is True
    assert "raciocínio" in ocorrencia.motivo


def test_saida_com_zero_tokens_e_vazio():
    ocorrencia = classificar(_resposta(texto="algo", tokens_saida=0), 100, Config())

    assert ocorrencia.tipo == TipoOcorrencia.VAZIO
    assert ocorrencia.alerta is True


def test_raciocinio_consome_o_teto():
    ocorrencia = classificar(
        _resposta(tokens_raciocinio=90),
        100,
        Config(limiar_raciocinio=0.9),
    )

    assert ocorrencia.tipo == TipoOcorrencia.LIMITE_CONSUMIDO_PELO_RACIOCINIO
    assert ocorrencia.alerta is True


def test_raciocinio_exatamente_no_limiar_dispara():
    ocorrencia = classificar(
        _resposta(tokens_raciocinio=90),
        100,
        Config(limiar_raciocinio=0.9),
    )

    assert ocorrencia.tipo == TipoOcorrencia.LIMITE_CONSUMIDO_PELO_RACIOCINIO


def test_motivo_de_parada_truncado():
    ocorrencia = classificar(
        _resposta(motivo_parada=MotivoParada.TRUNCADO),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.TRUNCADO
    assert ocorrencia.alerta is True


def test_saida_proxima_do_teto_sem_pontuacao_e_possivelmente_truncada():
    ocorrencia = classificar(
        _resposta(texto="resposta sem ponto final", tokens_saida=95),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.POSSIVELMENTE_TRUNCADO
    assert ocorrencia.alerta is True


def test_saida_exatamente_no_limiar_de_95_por_cento_dispara():
    ocorrencia = classificar(
        _resposta(texto="sem ponto final", tokens_saida=95),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.POSSIVELMENTE_TRUNCADO


def test_texto_com_pontuacao_final_nao_dispara():
    ocorrencia = classificar(
        _resposta(texto="resposta terminada.", tokens_saida=95),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.NENHUMA


def test_aspas_apos_a_pontuacao_contam_como_final():
    ocorrencia = classificar(
        _resposta(texto='Ele disse: "acabou."', tokens_saida=95),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.NENHUMA


def test_resposta_normal_nao_gera_ocorrencia():
    ocorrencia = classificar(_resposta(), 100, Config())

    assert ocorrencia.tipo == TipoOcorrencia.NENHUMA
    assert ocorrencia.alerta is False


def test_regra_1_tem_prioridade_sobre_regra_3():
    ocorrencia = classificar(
        _resposta(texto="", tokens_saida=5, tokens_raciocinio=100),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.VAZIO


def test_regra_2_tem_prioridade_sobre_regra_3():
    ocorrencia = classificar(
        _resposta(texto="algo", tokens_saida=0, tokens_raciocinio=100),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.VAZIO


def test_regra_3_tem_prioridade_sobre_regra_4():
    ocorrencia = classificar(
        _resposta(
            tokens_saida=10,
            tokens_raciocinio=100,
            motivo_parada=MotivoParada.TRUNCADO,
        ),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.LIMITE_CONSUMIDO_PELO_RACIOCINIO


def test_regra_4_tem_prioridade_sobre_regra_5():
    ocorrencia = classificar(
        _resposta(
            texto="sem ponto final",
            tokens_saida=100,
            motivo_parada=MotivoParada.TRUNCADO,
        ),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.TRUNCADO


def test_sem_limite_regras_3_e_5_nao_se_aplicam():
    ocorrencia = classificar(
        _resposta(
            texto="sem ponto final",
            tokens_saida=999,
            tokens_raciocinio=999,
        ),
        None,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.NENHUMA


def test_sem_limite_motivo_truncado_ainda_dispara_regra_4():
    ocorrencia = classificar(
        _resposta(motivo_parada=MotivoParada.TRUNCADO),
        None,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.TRUNCADO


def test_raciocinio_none_nao_dispara_regra_3():
    ocorrencia = classificar(
        _resposta(tokens_raciocinio=None),
        100,
        Config(),
    )

    assert ocorrencia.tipo == TipoOcorrencia.NENHUMA
