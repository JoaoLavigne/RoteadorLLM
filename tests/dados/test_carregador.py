import sqlite3

import pytest

from roteador.dados.carregador import (
    COLUNAS_DATASET,
    COLUNAS_OBRIGATORIAS,
    CarregadorCsv,
    CarregadorSqlite,
    LinhaDataset,
    Nota,
    carregar_dataset,
    criar_carregador,
    ler_dataset,
    salvar_dataset_sqlite,
)


def _escrever(tmp_path, linhas, cabecalho=None):
    cabecalho = cabecalho or COLUNAS_DATASET
    caminho = tmp_path / "dataset.csv"
    conteudo = [",".join(cabecalho), *linhas]
    caminho.write_text("\n".join(conteudo) + "\n", encoding="utf-8")
    return caminho


def _linha(**valores):
    padrao = {
        "id": "1",
        "documento": "doc-a",
        "tipo": "aberta",
        "pergunta": "Quanto é 2 + 2?",
        "nota_barato": "correta",
        "nota_forte": "correta",
        "tokens_resposta_barato": "10",
        "tokens_resposta_forte": "20",
        "contexto_tokens": "",
        "truncada_forte": "",
        "nota_forte_limitado": "",
        "tokens_resposta_forte_limitado": "",
    }
    padrao.update(valores)
    return ",".join(padrao[coluna] for coluna in COLUNAS_DATASET)


def test_carrega_todas_as_colunas(tmp_path):
    caminho = _escrever(
        tmp_path,
        [
            _linha(
                contexto_tokens="300",
                truncada_forte="true",
                nota_forte_limitado="parcial",
                tokens_resposta_forte_limitado="50",
            )
        ],
    )

    linhas = carregar_dataset(caminho)

    assert len(linhas) == 1
    assert linhas[0] == LinhaDataset(
        id="1",
        documento="doc-a",
        tipo="aberta",
        pergunta="Quanto é 2 + 2?",
        nota_barato=Nota.CORRETA,
        nota_forte=Nota.CORRETA,
        tokens_resposta_barato=10,
        tokens_resposta_forte=20,
        contexto_tokens=300,
        truncada_forte=True,
        nota_forte_limitado=Nota.PARCIAL,
        tokens_resposta_forte_limitado=50,
    )


def test_colunas_opcionais_vazias_recebem_padroes(tmp_path):
    caminho = _escrever(tmp_path, [_linha()])

    linha = carregar_dataset(caminho)[0]

    assert linha.contexto_tokens is None
    assert linha.truncada_forte is False
    assert linha.nota_forte_limitado is None
    assert linha.tokens_resposta_forte_limitado is None


def test_as_notas_sao_convertidas_para_os_valores_numericos(tmp_path):
    caminho = _escrever(
        tmp_path,
        [
            _linha(nota_barato="errada", nota_forte="parcial"),
            _linha(id="2", nota_barato="CORRETA", nota_forte="Errada"),
        ],
    )

    primeira, segunda = carregar_dataset(caminho)

    assert primeira.nota_barato == 0
    assert primeira.nota_forte == 1
    assert segunda.nota_barato == 2
    assert segunda.nota_forte == 0
    assert Nota.CORRETA.texto == "correta"


def test_nota_invalida_cita_o_numero_da_linha(tmp_path):
    caminho = _escrever(
        tmp_path,
        [_linha(), _linha(id="2", nota_barato="mais_ou_menos")],
    )

    with pytest.raises(ValueError, match="Linha 3"):
        carregar_dataset(caminho)


def test_inteiro_invalido_cita_o_numero_da_linha(tmp_path):
    caminho = _escrever(
        tmp_path,
        [_linha(tokens_resposta_forte="muitos")],
    )

    with pytest.raises(ValueError, match="Linha 2.*tokens_resposta_forte"):
        carregar_dataset(caminho)


def test_booleano_invalido_cita_o_numero_da_linha(tmp_path):
    caminho = _escrever(
        tmp_path,
        [_linha(truncada_forte="talvez")],
    )

    with pytest.raises(ValueError, match="Linha 2.*truncada_forte"):
        carregar_dataset(caminho)


def test_booleano_aceita_variacoes(tmp_path):
    caminho = _escrever(
        tmp_path,
        [
            _linha(truncada_forte="sim"),
            _linha(id="2", truncada_forte="1"),
            _linha(id="3", truncada_forte="false"),
            _linha(id="4", truncada_forte="0"),
        ],
    )

    linhas = carregar_dataset(caminho)

    assert [linha.truncada_forte for linha in linhas] == [True, True, False, False]


def test_cabecalho_sem_coluna_obrigatoria_falha(tmp_path):
    cabecalho = [c for c in COLUNAS_OBRIGATORIAS if c != "nota_forte"]
    caminho = _escrever(tmp_path, ["1,doc-a,aberta,pergunta,correta,10,20"], cabecalho)

    with pytest.raises(ValueError, match="nota_forte"):
        carregar_dataset(caminho)


def test_arquivo_inexistente_falha_com_mensagem_clara(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset não encontrado"):
        carregar_dataset(tmp_path / "nao_existe.csv")


def test_linhas_vazias_sao_ignoradas(tmp_path):
    caminho = _escrever(tmp_path, [_linha(), "", _linha(id="2")])

    assert [linha.id for linha in carregar_dataset(caminho)] == ["1", "2"]


def test_ler_dataset_e_alias_de_carregar_dataset(tmp_path):
    caminho = _escrever(tmp_path, [_linha()])

    assert ler_dataset(caminho) == carregar_dataset(caminho)


def _linha_tipada(**valores):
    padrao = {
        "id": "1",
        "documento": "doc-a",
        "tipo": "aberta",
        "pergunta": "pergunta",
        "nota_barato": Nota.CORRETA,
        "nota_forte": Nota.PARCIAL,
        "tokens_resposta_barato": 10,
        "tokens_resposta_forte": 20,
    }
    padrao.update(valores)
    return LinhaDataset(**padrao)


def _executar_sqlite(caminho, sql):
    with sqlite3.connect(caminho) as conexao:
        conexao.executescript(sql)


def test_salvar_e_carregar_sqlite_ida_e_volta(tmp_path):
    caminho = tmp_path / "dataset.db"
    linhas = [
        _linha_tipada(),
        _linha_tipada(
            id="2",
            documento="doc-b",
            nota_barato=Nota.ERRADA,
            nota_forte=Nota.CORRETA,
            contexto_tokens=300,
            truncada_forte=True,
            nota_forte_limitado=Nota.PARCIAL,
            tokens_resposta_forte_limitado=50,
        ),
    ]

    salvar_dataset_sqlite(linhas, caminho)

    assert carregar_dataset(caminho) == linhas


def test_carregar_dataset_detecta_sqlite_pela_extensao(tmp_path):
    caminho = tmp_path / "dataset.sqlite3"
    salvar_dataset_sqlite([_linha_tipada()], caminho)

    assert carregar_dataset(caminho) == [_linha_tipada()]


def test_criar_carregador_escolhe_pela_extensao(tmp_path):
    assert isinstance(criar_carregador("dados.csv"), CarregadorCsv)
    assert isinstance(criar_carregador("dados.db"), CarregadorSqlite)
    assert isinstance(criar_carregador("dados.sqlite"), CarregadorSqlite)
    assert isinstance(criar_carregador("dados.sqlite3"), CarregadorSqlite)
    assert isinstance(criar_carregador("dados.DB"), CarregadorSqlite)


def test_carregador_sqlite_usa_tabela_configuravel(tmp_path):
    caminho = tmp_path / "dataset.db"
    salvar_dataset_sqlite([_linha_tipada()], caminho, tabela="minha_tabela")

    linhas = CarregadorSqlite(caminho, tabela="minha_tabela").carregar()

    assert linhas == [_linha_tipada()]


def test_sqlite_sem_colunas_opcionais_usa_padroes(tmp_path):
    caminho = tmp_path / "dataset.db"
    _executar_sqlite(
        caminho,
        """
        CREATE TABLE dataset (
            id TEXT NOT NULL,
            documento TEXT NOT NULL,
            tipo TEXT NOT NULL,
            pergunta TEXT NOT NULL,
            nota_barato TEXT NOT NULL,
            nota_forte TEXT NOT NULL,
            tokens_resposta_barato INTEGER NOT NULL,
            tokens_resposta_forte INTEGER NOT NULL
        );
        INSERT INTO dataset VALUES
            ('1', 'doc-a', 'aberta', 'pergunta', 'correta', 'parcial', 10, 20);
        """,
    )

    linha = carregar_dataset(caminho)[0]

    assert linha.tokens_resposta_barato == 10
    assert linha.contexto_tokens is None
    assert linha.truncada_forte is False
    assert linha.nota_forte_limitado is None
    assert linha.tokens_resposta_forte_limitado is None


def test_sqlite_tabela_inexistente_falha_com_mensagem_clara(tmp_path):
    caminho = tmp_path / "dataset.db"
    sqlite3.connect(caminho).close()

    with pytest.raises(ValueError, match="Tabela 'dataset' não encontrada"):
        carregar_dataset(caminho)


def test_sqlite_sem_coluna_obrigatoria_falha(tmp_path):
    caminho = tmp_path / "dataset.db"
    _executar_sqlite(
        caminho,
        """
        CREATE TABLE dataset (
            id TEXT NOT NULL,
            documento TEXT NOT NULL
        );
        """,
    )

    with pytest.raises(ValueError, match="faltando.*nota_forte"):
        carregar_dataset(caminho)


def test_sqlite_com_nota_invalida_cita_o_registro(tmp_path):
    caminho = tmp_path / "dataset.db"
    _executar_sqlite(
        caminho,
        """
        CREATE TABLE dataset (
            id TEXT NOT NULL,
            documento TEXT NOT NULL,
            tipo TEXT NOT NULL,
            pergunta TEXT NOT NULL,
            nota_barato TEXT NOT NULL,
            nota_forte TEXT NOT NULL,
            tokens_resposta_barato INTEGER NOT NULL,
            tokens_resposta_forte INTEGER NOT NULL
        );
        INSERT INTO dataset VALUES
            ('1', 'doc-a', 'aberta', 'pergunta', 'talvez', 'correta', 10, 20);
        """,
    )

    with pytest.raises(ValueError, match="Registro 1.*nota_barato"):
        carregar_dataset(caminho)


def test_sqlite_arquivo_inexistente_falha_com_mensagem_clara(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset não encontrado"):
        carregar_dataset(tmp_path / "nao_existe.db")


def test_sqlite_arquivo_invalido_falha_com_mensagem_clara(tmp_path):
    caminho = tmp_path / "invalido.db"
    caminho.write_text("isto não é um banco sqlite", encoding="utf-8")

    with pytest.raises(ValueError, match="Não foi possível"):
        carregar_dataset(caminho)


def test_salvar_sqlite_substitui_o_conteudo_anterior(tmp_path):
    caminho = tmp_path / "dataset.db"
    salvar_dataset_sqlite([_linha_tipada(), _linha_tipada(id="2")], caminho)
    salvar_dataset_sqlite([_linha_tipada(id="3")], caminho)

    assert [linha.id for linha in carregar_dataset(caminho)] == ["3"]


def test_nome_de_tabela_invalido_falha(tmp_path):
    with pytest.raises(ValueError, match="Nome de tabela inválido"):
        CarregadorSqlite(tmp_path / "dataset.db", tabela="dataset; DROP TABLE x")

    with pytest.raises(ValueError, match="Nome de tabela inválido"):
        salvar_dataset_sqlite([], tmp_path / "dataset.db", tabela="1tabela")
