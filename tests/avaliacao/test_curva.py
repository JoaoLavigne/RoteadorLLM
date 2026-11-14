import csv

import pytest

from roteador.avaliacao.curva import (
    COLUNAS_CURVA,
    frase_resultado,
    gerar_curva,
    linhas_curva,
    salvar_curva,
    tabela_markdown,
)
from roteador.avaliacao.politicas import ResultadoPolitica, avaliar_politicas
from roteador.config import Config
from roteador.dados.carregador import LinhaDataset, Nota


class TokenizadorFixo:
    def contar(self, texto):
        return 10


def _cfg():
    return Config(
        preco_ref={
            "barato": (1_000_000.0, 2_000_000.0),
            "forte": (4_000_000.0, 8_000_000.0),
        }
    )


def _resultado(**valores):
    padrao = {
        "nome": "politica",
        "qualidade_media": 0.625,
        "custo_estimado_usd": 140.0,
        "custo_medio_usd": 35.0,
        "qualidade_relativa_pct": 83.333333,
        "custo_relativo_pct": 5.0,
        "economia_pct": 95.0,
        "linhas": 4,
    }
    padrao.update(valores)
    return ResultadoPolitica(**padrao)


def _linha(**valores):
    padrao = {
        "id": "1",
        "documento": "doc-a",
        "tipo": "aberta",
        "pergunta": "pergunta",
        "nota_barato": Nota.CORRETA,
        "nota_forte": Nota.CORRETA,
        "tokens_resposta_barato": 10,
        "tokens_resposta_forte": 100,
    }
    padrao.update(valores)
    return LinhaDataset(**padrao)


def test_frase_resultado_no_formato_do_projeto():
    assert frase_resultado(_resultado()) == (
        "Mantive 83.3% da qualidade do forte gastando 95.0% menos."
    )


def test_frase_resultado_quando_fica_mais_caro():
    resultado = _resultado(economia_pct=-10.0)

    assert frase_resultado(resultado) == (
        "Mantive 83.3% da qualidade do forte gastando 10.0% mais."
    )


def test_tabela_markdown_tem_cabecalho_politica_e_frase():
    tabela = tabela_markdown([_resultado(nome="sempre_barato")])

    assert "| Política |" in tabela
    assert "sempre_barato" in tabela
    assert "83.3%" in tabela
    assert "95.0%" in tabela
    assert "Mantive 83.3% da qualidade do forte" in tabela


def test_linhas_curva_ignora_politicas_indisponiveis():
    disponivel = _resultado(nome="sempre_barato")
    indisponivel = _resultado(
        nome="forte_limitado", aviso="Política ignorada."
    )

    linhas = linhas_curva([disponivel, indisponivel])

    assert [linha["politica"] for linha in linhas] == ["sempre_barato"]
    assert set(linhas[0]) == set(COLUNAS_CURVA)


def test_linhas_curva_sem_politicas_disponiveis_falha():
    with pytest.raises(ValueError, match="Nenhuma política"):
        linhas_curva([_resultado(aviso="ignorada")])


def test_salvar_curva_grava_csv_com_cabecalho(tmp_path):
    caminho = tmp_path / "curva.csv"

    linhas = salvar_curva(
        [_resultado(nome="sempre_barato"), _resultado(nome="perfeito")], caminho
    )

    assert len(linhas) == 2
    with caminho.open(encoding="utf-8", newline="") as aberto:
        registros = list(csv.DictReader(aberto))
    assert list(registros[0]) == list(COLUNAS_CURVA)
    assert [registro["politica"] for registro in registros] == [
        "sempre_barato",
        "perfeito",
    ]


def test_gerar_curva_grava_csv_e_devolve_markdown(tmp_path):
    caminho = tmp_path / "curva.csv"
    resultados = [
        _resultado(nome="sempre_barato"),
        _resultado(nome="sempre_forte", qualidade_relativa_pct=100.0,
                   custo_relativo_pct=100.0, economia_pct=0.0),
    ]

    tabela = gerar_curva(resultados, caminho)

    assert caminho.exists()
    assert "sempre_forte" in tabela
    assert "Mantive 100.0% da qualidade do forte gastando 0.0% menos." in tabela


def test_curva_sai_de_um_dataset_de_exemplo(tmp_path):
    linhas = [
        _linha(id="1"),
        _linha(id="2", nota_barato=Nota.ERRADA, nota_forte=Nota.CORRETA),
    ]
    resultados = avaliar_politicas(linhas, _cfg(), tokenizador=TokenizadorFixo())
    caminho = tmp_path / "curva.csv"

    tabela = gerar_curva(resultados, caminho)

    assert caminho.exists()
    assert "sempre_barato" in tabela
    assert "sempre_forte" in tabela
    assert "perfeito" in tabela
    assert "v1" in tabela
