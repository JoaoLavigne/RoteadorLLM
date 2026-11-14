import csv
from pathlib import Path

from roteador.adapters.falso import AdapterFalso
from roteador.cli import COLUNAS_COLETA, main
from roteador.config import Config
from roteador.dados.carregador import COLUNAS_OBRIGATORIAS
from roteador.ml.classificadores import carregar_modelo
from roteador.ml.embedder import EmbedderFalso
from roteador.registro import ler_csv
from roteador.tipos import MotivoParada, Papel, RespostaModelo


class TokenizadorFixo:
    def __init__(self, tokens=1):
        self.tokens = tokens

    def contar(self, texto):
        return self.tokens


def _cfg(tmp_path, **extra):
    valores = {
        "caminho_registro": str(tmp_path / "resultados" / "registro.csv"),
        "caminho_ml": str(tmp_path / "resultados" / "ml_ocorrencias.csv"),
        "caminho_dataset": str(tmp_path / "dataset.csv"),
        "caminho_modelo_v2": str(tmp_path / "resultados" / "modelo_v2.joblib"),
        "limiar_tokens_pergunta": 100,
        "janela_contexto_barato": 1000,
        "tentativas_comunicacao": 1,
        "fallback_comunicacao": False,
        "espera_entre_tentativas_s": 0,
        "preco_ref": {
            "barato": (1_000_000.0, 2_000_000.0),
            "forte": (4_000_000.0, 8_000_000.0),
        },
    }
    valores.update(extra)
    return Config(**valores)


def _resposta(texto="ok", modelo="falso", tokens_saida=10):
    return RespostaModelo(
        texto=texto,
        tokens_entrada=5,
        tokens_saida=tokens_saida,
        tokens_raciocinio=None,
        motivo_parada=MotivoParada.COMPLETO,
        tempo_ms=12.5,
        modelo=modelo,
    )


def _escrever_dataset(caminho, linhas=None):
    linhas = linhas or [
        "1,doc-a,aberta,pergunta 1,correta,correta,10,100",
        "2,doc-a,aberta,pergunta 2,errada,correta,10,100",
        "3,doc-b,aberta,pergunta 3,parcial,parcial,20,80",
    ]
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        ",".join(COLUNAS_OBRIGATORIAS) + "\n" + "\n".join(linhas) + "\n",
        encoding="utf-8",
    )
    return caminho


def _escrever_registros(cfg, registro, ml):
    Path(cfg.caminho_registro).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.caminho_registro).write_text(registro, encoding="utf-8")
    Path(cfg.caminho_ml).write_text(ml, encoding="utf-8")


def test_rotear_mostra_decisao_resposta_e_custo(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    barato = AdapterFalso([_resposta("resposta barata", "barato")], nome="barato")
    forte = AdapterFalso([_resposta("resposta forte", "forte")], nome="forte")

    codigo = main(
        ["rotear", "pergunta"],
        cfg=cfg,
        adapters={Papel.BARATO: barato, Papel.FORTE: forte},
        tokenizador=TokenizadorFixo(1),
    )

    saida = capsys.readouterr().out
    assert codigo == 0
    assert "Papel: barato" in saida
    assert "resposta barata" in saida
    assert "Custo estimado" in saida
    registros = ler_csv(cfg.caminho_registro)
    assert len(registros) == 1
    assert registros[0]["papel"] == "barato"


def test_rotear_le_o_contexto_do_arquivo(tmp_path):
    cfg = _cfg(tmp_path)
    contexto = tmp_path / "contexto.txt"
    contexto.write_text("conteúdo do contexto", encoding="utf-8")
    barato = AdapterFalso([_resposta()], nome="barato")

    codigo = main(
        ["rotear", "pergunta", "--contexto", str(contexto)],
        cfg=cfg,
        adapters={Papel.BARATO: barato},
        tokenizador=TokenizadorFixo(1),
    )

    assert codigo == 0
    assert "conteúdo do contexto" in barato.chamadas[0].prompt


def test_rotear_sem_chave_do_gemini_mostra_erro(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    cfg = _cfg(tmp_path, limiar_tokens_pergunta=0, fallback_comunicacao=False)

    codigo = main(
        ["rotear", "pergunta"],
        cfg=cfg,
        tokenizador=TokenizadorFixo(1),
    )

    erro = capsys.readouterr().err
    assert codigo == 1
    assert "Gemini" in erro or "GEMINI" in erro


def test_coletar_barato_grava_csv_com_nota_em_branco(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    _escrever_dataset(Path(cfg.caminho_dataset))
    barato = AdapterFalso(
        [
            _resposta("resposta 1", "barato"),
            _resposta("resposta 2", "barato"),
            _resposta("resposta 3", "barato"),
        ],
        nome="barato",
    )

    codigo = main(
        ["coletar", "--politica", "barato"],
        cfg=cfg,
        adapters={Papel.BARATO: barato},
        tokenizador=TokenizadorFixo(1),
    )

    saida = capsys.readouterr().out
    assert codigo == 0
    assert "3 resposta(s) gravada(s)" in saida

    caminho = Path(cfg.caminho_registro).parent / "coleta_barato.csv"
    with caminho.open(encoding="utf-8", newline="") as aberto:
        registros = list(csv.DictReader(aberto))

    assert list(registros[0]) == list(COLUNAS_COLETA)
    assert len(registros) == 3
    assert registros[0]["resposta"] == "resposta 1"
    assert registros[0]["politica"] == "barato"
    assert registros[0]["nota"] == ""
    assert registros[0]["tokens_pergunta"] == "1"
    assert len(ler_csv(cfg.caminho_registro)) == 3


def test_coletar_com_dataset_inexistente_mostra_erro(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    caminho = tmp_path / "nao_existe.csv"

    codigo = main(
        ["coletar", "--politica", "barato", "--dataset", str(caminho)],
        cfg=cfg,
        adapters={},
        tokenizador=TokenizadorFixo(1),
    )

    erro = capsys.readouterr().err
    assert codigo == 1
    assert "Dataset não encontrado" in erro


def test_coletar_forte_limitado_sem_x_falha(tmp_path, capsys):
    cfg = _cfg(tmp_path, x_tokens_forte_limitado=None)
    _escrever_dataset(Path(cfg.caminho_dataset))

    codigo = main(
        ["coletar", "--politica", "forte_limitado"],
        cfg=cfg,
        adapters={Papel.FORTE: AdapterFalso([_resposta()])},
        tokenizador=TokenizadorFixo(1),
    )

    erro = capsys.readouterr().err
    assert codigo == 1
    assert "x_tokens_forte_limitado" in erro


def test_coletar_forte_limitado_aplica_o_limite(tmp_path):
    cfg = _cfg(tmp_path, x_tokens_forte_limitado=50)
    _escrever_dataset(Path(cfg.caminho_dataset))
    forte = AdapterFalso(
        [
            _resposta("resposta limitada 1", "forte"),
            _resposta("resposta limitada 2", "forte"),
            _resposta("resposta limitada 3", "forte"),
        ],
        nome="forte",
    )

    codigo = main(
        ["coletar", "--politica", "forte_limitado"],
        cfg=cfg,
        adapters={Papel.FORTE: forte},
        tokenizador=TokenizadorFixo(1),
    )

    assert codigo == 0
    assert forte.chamadas[0].limite_tokens == 75
    assert "Limite seus gastos a cerca de 50 tokens." in forte.chamadas[0].prompt


def test_avaliar_gera_curva_e_relatorio_de_alertas(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    _escrever_dataset(Path(cfg.caminho_dataset))
    _escrever_registros(
        cfg,
        "modelo,faixa_prevista\nbarato,curta\nbarato,curta\n",
        "modelo,faixa_prevista,tipo_ocorrencia\nbarato,curta,vazio\n",
    )

    codigo = main(
        ["avaliar"],
        cfg=cfg,
        tokenizador=TokenizadorFixo(1),
    )

    saida = capsys.readouterr()
    assert codigo == 0
    assert "sempre_barato" in saida.out
    assert "sempre_forte" in saida.out
    assert "Curva gravada em" in saida.out
    assert "Relatório de alertas:" in saida.out
    assert "50.0%" in saida.out
    assert "forte_limitado" in saida.err
    assert (Path(cfg.caminho_registro).parent / "curva.csv").exists()


def test_avaliar_com_dataset_inexistente_mostra_erro(tmp_path, capsys):
    cfg = _cfg(tmp_path)

    codigo = main(
        ["avaliar"],
        cfg=cfg,
        tokenizador=TokenizadorFixo(1),
    )

    erro = capsys.readouterr().err
    assert codigo == 1
    assert "Dataset não encontrado" in erro


def test_treinar_grava_modelo_v2(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    _escrever_dataset(Path(cfg.caminho_dataset))
    embedder = EmbedderFalso(dimensao=4)

    codigo = main(
        ["treinar"],
        cfg=cfg,
        tokenizador=TokenizadorFixo(1),
        embedder=embedder,
    )

    saida = capsys.readouterr().out
    assert codigo == 0
    assert "Modelo v2 treinado" in saida
    assert "Acurácia do classificador A" in saida
    assert "Acurácia do classificador B" in saida

    caminho = Path(cfg.caminho_modelo_v2)
    assert caminho.exists()
    modelo = carregar_modelo(caminho, embedder_esperado=embedder)
    assert modelo.embedder_nome == embedder.nome


def test_coletar_v2_usa_o_modelo_treinado(tmp_path):
    cfg = _cfg(tmp_path)
    _escrever_dataset(Path(cfg.caminho_dataset))
    embedder = EmbedderFalso(dimensao=4)
    assert (
        main(
            ["treinar"],
            cfg=cfg,
            tokenizador=TokenizadorFixo(1),
            embedder=embedder,
        )
        == 0
    )
    barato = AdapterFalso(
        [_resposta("barato", "barato") for _ in range(3)], nome="barato"
    )
    forte = AdapterFalso(
        [_resposta("forte", "forte") for _ in range(3)], nome="forte"
    )

    codigo = main(
        ["coletar", "--politica", "v2"],
        cfg=cfg,
        adapters={Papel.BARATO: barato, Papel.FORTE: forte},
        tokenizador=TokenizadorFixo(1),
        embedder=embedder,
    )

    assert codigo == 0
    caminho = Path(cfg.caminho_registro).parent / "coleta_v2.csv"
    with caminho.open(encoding="utf-8", newline="") as aberto:
        registros = list(csv.DictReader(aberto))
    assert len(registros) == 3
    assert all(registro["politica"] == "v2" for registro in registros)


def test_coletar_v2_sem_modelo_mostra_erro(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    _escrever_dataset(Path(cfg.caminho_dataset))

    codigo = main(
        ["coletar", "--politica", "v2"],
        cfg=cfg,
        adapters={},
        tokenizador=TokenizadorFixo(1),
        embedder=EmbedderFalso(dimensao=4),
    )

    erro = capsys.readouterr().err
    assert codigo == 1
    assert "Modelo v2 não encontrado" in erro


def test_avaliar_inclui_a_politica_v2_quando_ha_modelo(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    _escrever_dataset(Path(cfg.caminho_dataset))
    embedder = EmbedderFalso(dimensao=4)
    assert (
        main(
            ["treinar"],
            cfg=cfg,
            tokenizador=TokenizadorFixo(1),
            embedder=embedder,
        )
        == 0
    )
    capsys.readouterr()

    codigo = main(
        ["avaliar"],
        cfg=cfg,
        tokenizador=TokenizadorFixo(1),
        embedder=embedder,
    )

    saida = capsys.readouterr()
    assert codigo == 0
    assert "| v2 |" in saida.out
    assert "Política v2 ignorada" not in saida.err
