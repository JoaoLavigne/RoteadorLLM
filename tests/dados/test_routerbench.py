import pandas as pd
import pytest

from roteador.dados.carregador import Nota
from roteador.dados.rotulos import barato_basta
from roteador.dados.routerbench import (
    carregar_dataframe,
    converter_para_dataset,
    escolher_par_mais_barato,
    modelos_disponiveis,
    normalizar_prompt,
    nota_de_score,
    resumo_par,
    selecionar_recorte,
)


class TokenizadorFixo:
    def __init__(self, tokens):
        self.tokens = tokens

    def contar(self, texto):
        return self.tokens


def _dados(**extra):
    valores = {
        "sample_id": ["a.0", "a.1", "b.0", "b.1"],
        "prompt": ["p0", "p1", "p2", "p3"],
        "eval_name": ["mmlu", "mmlu", "gsm", "gsm"],
        "barato": [1.0, 0.5, 0.0, 1.0],
        "barato|model_response": ["certa", "meio", "errada", "certa"],
        "barato|total_cost": [0.10, 0.20, 0.30, 0.40],
        "forte": [1.0, 1.0, 0.0, 0.5],
        "forte|model_response": ["certa", "certa", "errada", "meio"],
        "forte|total_cost": [1.0, 2.0, 3.0, 4.0],
    }
    valores.update(extra)
    return pd.DataFrame(valores)


def test_converte_as_colunas_para_linha_dataset():
    linha = converter_para_dataset(_dados(), "barato", "forte")[0]

    assert linha.id == "a.0"
    assert linha.documento == "p0"
    assert linha.tipo == "mmlu"
    assert linha.pergunta == "p0"
    assert linha.nota_barato == Nota.CORRETA
    assert linha.nota_forte == Nota.CORRETA
    assert linha.score_barato == pytest.approx(1.0)
    assert linha.score_forte == pytest.approx(1.0)


def test_score_continuo_vira_correta_parcial_ou_errada():
    linhas = converter_para_dataset(_dados(), "barato", "forte")

    assert [linha.nota_barato for linha in linhas] == [
        Nota.CORRETA,
        Nota.PARCIAL,
        Nota.ERRADA,
        Nota.CORRETA,
    ]
    assert [linha.nota_forte for linha in linhas] == [
        Nota.CORRETA,
        Nota.CORRETA,
        Nota.ERRADA,
        Nota.PARCIAL,
    ]


def test_regra_barato_basta_funciona_na_linha_convertida():
    linhas = converter_para_dataset(_dados(), "barato", "forte")

    assert barato_basta(linhas[0]) is True
    assert barato_basta(linhas[1]) is False
    assert barato_basta(linhas[2]) is True
    assert barato_basta(linhas[3]) is True


def test_score_bruto_decide_o_barato_basta_apesar_das_notas():
    dados = _dados()
    dados.loc[0, "barato"] = 0.5
    dados.loc[0, "forte"] = 0.75

    linha = converter_para_dataset(dados, "barato", "forte")[0]

    assert linha.nota_barato == Nota.PARCIAL
    assert linha.nota_forte == Nota.PARCIAL
    assert linha.score_barato == pytest.approx(0.5)
    assert linha.score_forte == pytest.approx(0.75)
    assert barato_basta(linha) is False


def test_tokens_de_resposta_usam_o_tokenizador_informado():
    linhas = converter_para_dataset(
        _dados(), "barato", "forte", tokenizador=TokenizadorFixo(7)
    )

    assert [linha.tokens_resposta_barato for linha in linhas] == [7, 7, 7, 7]
    assert [linha.tokens_resposta_forte for linha in linhas] == [7, 7, 7, 7]


def test_sem_tokenizador_aproxima_pelo_divisor():
    dados = _dados()
    dados["barato|model_response"] = ["abcde", "abcd", "", "a"]

    linhas = converter_para_dataset(
        dados, "barato", "forte", divisor_aproximado=2
    )

    assert [linha.tokens_resposta_barato for linha in linhas] == [3, 2, 0, 1]


def test_linhas_sem_score_sao_ignoradas():
    dados = _dados()
    dados.loc[1, "forte"] = None

    linhas = converter_para_dataset(dados, "barato", "forte")

    assert [linha.id for linha in linhas] == ["a.0", "b.0", "b.1"]


def test_sem_linhas_validas_falha():
    dados = _dados()
    dados["forte"] = [None, None, None, None]

    with pytest.raises(ValueError, match="Nenhuma linha válida"):
        converter_para_dataset(dados, "barato", "forte")


def test_modelo_sem_colunas_cita_o_nome():
    with pytest.raises(ValueError, match="Modelo 'ausente' incompleto"):
        converter_para_dataset(_dados(), "ausente", "forte")


def test_coluna_obrigatoria_ausente_falha():
    dados = _dados().drop(columns=["eval_name"])

    with pytest.raises(ValueError, match="eval_name"):
        converter_para_dataset(dados, "barato", "forte")


def test_campo_de_texto_vazio_cita_a_linha():
    dados = _dados()
    dados.loc[2, "prompt"] = ""

    with pytest.raises(ValueError, match="Linha 2.*prompt"):
        converter_para_dataset(dados, "barato", "forte")


def test_divisor_invalido_falha():
    with pytest.raises(ValueError, match="divisor aproximado"):
        converter_para_dataset(
            _dados(), "barato", "forte", divisor_aproximado=0
        )


def test_prompt_em_lista_vira_texto_simples():
    valor = "['primeira mensagem', 'segunda mensagem']"

    assert normalizar_prompt(valor) == "primeira mensagem\n\nsegunda mensagem"


def test_prompt_simples_e_lista_invalida_ficam_como_estao():
    assert normalizar_prompt("pergunta direta") == "pergunta direta"
    assert normalizar_prompt("[isto não é uma lista") == "[isto não é uma lista"


def test_conversao_normaliza_o_prompt_da_lista():
    dados = _dados()
    dados.loc[0, "prompt"] = "['qual é a resposta?']"

    linha = converter_para_dataset(dados, "barato", "forte")[0]

    assert linha.pergunta == "qual é a resposta?"
    assert linha.documento == "qual é a resposta?"


def test_nota_de_score_valida_as_bordas():
    assert nota_de_score(0) == Nota.ERRADA
    assert nota_de_score(0.25) == Nota.PARCIAL
    assert nota_de_score(1.0) == Nota.CORRETA


def test_nota_de_score_rejeita_valores_invalidos():
    for valor in (-0.1, float("nan"), True, "1"):
        with pytest.raises(ValueError, match="Score inválido"):
            nota_de_score(valor)


def test_modelos_disponiveis_so_lista_modelos_completos():
    dados = _dados()
    dados["incompleto"] = [0.0, 0.0, 0.0, 0.0]

    assert modelos_disponiveis(dados) == ["barato", "forte"]
    assert "incompleto" not in modelos_disponiveis(dados)


def test_recorte_e_reprodutivel_e_tem_o_tamanho_pedido():
    dados = _dados()

    primeiro = selecionar_recorte(dados, 2, semente=7)
    segundo = selecionar_recorte(dados, 2, semente=7)

    assert len(primeiro) == 2
    assert primeiro.equals(segundo)


def test_recorte_invalido_falha():
    with pytest.raises(ValueError, match="positiva"):
        selecionar_recorte(_dados(), 0)

    with pytest.raises(ValueError, match="supera"):
        selecionar_recorte(_dados(), 10)


def test_carregar_csv_e_pickle(tmp_path):
    dados = _dados()
    csv = tmp_path / "recorte.csv"
    pkl = tmp_path / "recorte.pkl"
    dados.to_csv(csv, index=False)
    dados.to_pickle(pkl)

    assert len(carregar_dataframe(csv)) == len(dados)
    assert len(carregar_dataframe(pkl)) == len(dados)


def test_carregar_arquivo_inexistente_falha(tmp_path):
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        carregar_dataframe(tmp_path / "nao_existe.pkl")


def test_carregar_extensao_nao_suportada_falha(tmp_path):
    caminho = tmp_path / "recorte.txt"
    caminho.write_text("texto", encoding="utf-8")

    with pytest.raises(ValueError, match="Extensão não suportada"):
        carregar_dataframe(caminho)


def test_escolhe_o_barato_mais_barato_que_atinge_a_meta():
    dados = pd.DataFrame(
        {
            "sample_id": ["1", "2"],
            "prompt": ["p0", "p1"],
            "eval_name": ["mmlu", "mmlu"],
            "barato": [1.0, 1.0],
            "barato|model_response": ["a", "b"],
            "barato|total_cost": [0.1, 0.1],
            "outro": [1.0, 1.0],
            "outro|model_response": ["a", "b"],
            "outro|total_cost": [0.3, 0.3],
            "forte": [1.0, 1.0],
            "forte|model_response": ["a", "b"],
            "forte|total_cost": [1.0, 1.0],
        }
    )

    assert escolher_par_mais_barato(dados, meta_qualidade=0.95) == (
        "barato",
        "forte",
    )


def test_sem_par_que_atinge_a_meta_devolve_none():
    dados = pd.DataFrame(
        {
            "sample_id": ["1", "2"],
            "prompt": ["p0", "p1"],
            "eval_name": ["mmlu", "mmlu"],
            "barato": [0.0, 0.0],
            "barato|model_response": ["a", "b"],
            "barato|total_cost": [0.1, 0.1],
            "forte": [1.0, 1.0],
            "forte|model_response": ["a", "b"],
            "forte|total_cost": [1.0, 1.0],
        }
    )

    assert escolher_par_mais_barato(dados, meta_qualidade=0.95) is None


def test_resumo_par_traz_qualidade_e_custo_medios():
    resumo = resumo_par(_dados(), "barato", "forte")

    assert resumo["barato"]["qualidade_media"] == pytest.approx(0.625)
    assert resumo["barato"]["custo_medio_usd"] == pytest.approx(0.25)
    assert resumo["forte"]["qualidade_media"] == pytest.approx(0.625)
    assert resumo["forte"]["custo_medio_usd"] == pytest.approx(2.5)
