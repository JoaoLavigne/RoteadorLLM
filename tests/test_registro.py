import pytest

from roteador.registro import (
    COLUNAS_ML,
    COLUNAS_REGISTRO,
    Registro,
    RegistroCsv,
    RegistroMemoria,
    ler_csv,
)


def _linha_registro(**extra):
    linha = {coluna: f"valor_{coluna}" for coluna in COLUNAS_REGISTRO}
    linha.update(extra)
    return linha


def _linha_ml(**extra):
    linha = {coluna: f"valor_{coluna}" for coluna in COLUNAS_ML}
    linha.update(extra)
    return linha


def _cabecalho(colunas):
    return ",".join(colunas)


def test_cabecalho_criado_uma_vez(tmp_path):
    caminho = tmp_path / "registro.csv"

    RegistroCsv(caminho, COLUNAS_REGISTRO)
    RegistroCsv(caminho, COLUNAS_REGISTRO)

    linhas = caminho.read_text(encoding="utf-8").splitlines()
    assert linhas == [_cabecalho(COLUNAS_REGISTRO)]


def test_varias_linhas_sem_duplicar_cabecalho(tmp_path):
    caminho = tmp_path / "registro.csv"
    registro = RegistroCsv(caminho, COLUNAS_REGISTRO)

    registro.adicionar(_linha_registro(id_requisicao="1"))
    registro.adicionar(_linha_registro(id_requisicao="2"))

    linhas = caminho.read_text(encoding="utf-8").splitlines()
    assert len(linhas) == 3
    assert linhas[0] == _cabecalho(COLUNAS_REGISTRO)
    assert linhas.count(_cabecalho(COLUNAS_REGISTRO)) == 1


def test_campo_faltando_gera_erro(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    linha = _linha_registro()
    del linha["modelo"]

    with pytest.raises(ValueError, match="faltando.*modelo"):
        registro.adicionar(linha)


def test_campo_sobrando_gera_erro(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    linha = _linha_registro()
    linha["campo_extra"] = "sobrou"

    with pytest.raises(ValueError, match="sobrando.*campo_extra"):
        registro.adicionar(linha)


def test_linha_invalida_nao_grava_nada(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    linha = _linha_registro()
    del linha["modelo"]

    with pytest.raises(ValueError):
        registro.adicionar(linha)

    assert registro.ler() == []


def test_leitura_devolve_o_que_foi_gravado_com_virgulas_aspas_e_acentos(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    linha = _linha_registro(
        modelo="llama3.2:3b — ação",
        motivo_decisao='pergunta acima do limiar, com vírgula',
        ocorrencia='resposta "cortada"',
    )

    registro.adicionar(linha)

    assert registro.ler() == [linha]


def test_valores_numericos_e_none_viram_texto_no_csv(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    linha = _linha_registro(tokens_pergunta=12, tokens_saida=None)

    registro.adicionar(linha)
    lido = registro.ler()[0]

    assert lido["tokens_pergunta"] == "12"
    assert lido["tokens_saida"] == ""


def test_dois_registros_usam_esquemas_diferentes(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    registro_ml = RegistroCsv(tmp_path / "ml.csv", COLUNAS_ML)

    registro.adicionar(_linha_registro())
    registro_ml.adicionar(_linha_ml())

    assert list(registro.ler()[0]) == list(COLUNAS_REGISTRO)
    assert list(registro_ml.ler()[0]) == list(COLUNAS_ML)
    assert registro.caminho.read_text(encoding="utf-8").splitlines()[0] == _cabecalho(
        COLUNAS_REGISTRO
    )
    assert registro_ml.caminho.read_text(encoding="utf-8").splitlines()[0] == _cabecalho(
        COLUNAS_ML
    )


def test_ler_arquivo_inexistente_devolve_lista_vazia(tmp_path):
    assert ler_csv(tmp_path / "nao_existe.csv") == []


def test_cria_diretorio_pai(tmp_path):
    caminho = tmp_path / "subpasta" / "registro.csv"

    RegistroCsv(caminho, COLUNAS_REGISTRO)

    assert caminho.exists()
    assert caminho.read_text(encoding="utf-8").splitlines() == [
        _cabecalho(COLUNAS_REGISTRO)
    ]


def test_registro_csv_e_registro_memoria_cumprem_a_interface(tmp_path):
    assert isinstance(RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO), Registro)
    assert isinstance(RegistroMemoria(COLUNAS_REGISTRO), Registro)


def test_registro_memoria_grava_e_le_como_texto():
    registro = RegistroMemoria(COLUNAS_REGISTRO)
    linha = _linha_registro(tokens_pergunta=12, tokens_saida=None)

    registro.adicionar(linha)
    lido = registro.ler()[0]

    assert list(lido) == list(COLUNAS_REGISTRO)
    assert lido["tokens_pergunta"] == "12"
    assert lido["tokens_saida"] == ""


def test_registro_memoria_valida_colunas():
    registro = RegistroMemoria(COLUNAS_REGISTRO)
    linha = _linha_registro()
    del linha["modelo"]

    with pytest.raises(ValueError, match="faltando.*modelo"):
        registro.adicionar(linha)

    assert registro.ler() == []


def test_mapeamento_de_todas_as_colunas_nao_desloca_valores(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    linha = {coluna: f"<{coluna}>" for coluna in COLUNAS_REGISTRO}

    registro.adicionar(linha)

    assert registro.ler() == [linha]


def test_round_trip_preserva_caracteres_especiais(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    especial = 'ação, ção; "aspas"; barra | tab\tquebra\nlinha 2 \\ fim 😀'
    linha = _linha_registro(
        modelo=especial,
        motivo_decisao=especial,
        ocorrencia=especial,
    )

    registro.adicionar(linha)

    assert registro.ler() == [linha]


def test_campos_vazios_e_nulos_viram_texto_vazio(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)
    linha = _linha_registro(
        modelo="",
        motivo_decisao=None,
        tokens_pergunta=0,
        tokens_saida="",
    )

    registro.adicionar(linha)
    lido = registro.ler()[0]

    assert lido["modelo"] == ""
    assert lido["motivo_decisao"] == ""
    assert lido["tokens_pergunta"] == "0"
    assert lido["tokens_saida"] == ""
    assert lido["ocorrencia"] == "valor_ocorrencia"


def test_linha_que_nao_e_mapeamento_gera_erro(tmp_path):
    registro = RegistroCsv(tmp_path / "registro.csv", COLUNAS_REGISTRO)

    with pytest.raises(TypeError, match="mapeamento"):
        registro.adicionar(["modelo", "barato"])

    assert registro.ler() == []


def test_registro_memoria_preserva_especiais_vazios_e_nulos():
    registro = RegistroMemoria(COLUNAS_REGISTRO)
    especial = 'ação, ção; "aspas"; barra | tab\tquebra\nlinha 2 \\ fim 😀'
    linha = _linha_registro(
        modelo=especial,
        motivo_decisao=None,
        tokens_pergunta=0,
    )

    registro.adicionar(linha)
    lido = registro.ler()[0]

    assert lido["modelo"] == especial
    assert lido["motivo_decisao"] == ""
    assert lido["tokens_pergunta"] == "0"
    assert list(lido) == list(COLUNAS_REGISTRO)


def test_registro_memoria_nao_grava_linha_que_nao_e_mapeamento():
    registro = RegistroMemoria(COLUNAS_REGISTRO)

    with pytest.raises(TypeError, match="mapeamento"):
        registro.adicionar(["modelo", "barato"])

    assert registro.ler() == []
