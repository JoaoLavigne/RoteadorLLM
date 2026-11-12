# Roteador de LLMs (custo × qualidade)

Roteador que decide, para cada pergunta, se responde com o modelo **barato** (local, `llama3.2:3b` via Ollama) ou com o modelo **forte** (Gemini Flash). Cada chamada é registrada e o sistema é avaliado por uma curva de custo × qualidade.

O objetivo é mostrar, com números, quanto custo é economizado mantendo a maior parte da qualidade do modelo forte.

## Como funciona

1. O **tokenizador** conta os tokens da pergunta (e do contexto, quando houver).
2. O **roteador** devolve uma `Decisao`: papel (barato ou forte), motivo, faixa prevista e limite de tokens.
3. Se houver limite `X`, o prompt recebe o aviso "Limite seus gastos a cerca de X tokens." e um teto rígido é aplicado.
4. O **adapter** do papel escolhido gera a resposta.
5. O **detector** classifica a resposta como completa, vazia ou truncada.
6. Respostas vazias ou truncadas recebem **uma única nova tentativa**: o forte repete com o teto dobrado; o barato escala para o forte.
7. O custo estimado é calculado e a requisição é gravada no registro. Ocorrências também vão para o arquivo de ML.
8. O orquestrador devolve um `Resultado`.

### Roteador v1 e v2

- **v1 (determinístico):** decide apenas pela contagem de tokens da pergunta e por uma regra de janela de contexto, sem lista de palavras-chave.
- **v2 (ML):** dois classificadores sobre o embedding da pergunta e a contagem de tokens:
  - **A (qualidade):** "o modelo barato basta?" — decide o roteamento;
  - **B (tamanho):** prevê a faixa da resposta (curta, média ou longa) e define o limite de tokens no forte.

## Arquitetura

```
.
├── pyproject.toml
├── src/roteador/
│   ├── tipos.py              dataclasses, enums e interfaces (Protocol)
│   ├── config.py             todos os valores ajustáveis
│   ├── tokenizador.py        contagem aproximada de tokens
│   ├── adapters/             base.py, falso.py, ollama.py, gemini.py
│   ├── detector.py           classifica respostas vazias ou truncadas
│   ├── registro.py           CSV de requisições e arquivo de ML
│   ├── limite.py             prompt de limite e teto rígido
│   ├── roteador_v1.py
│   ├── orquestrador.py       junta tudo e trata a nova tentativa
│   ├── dados/                carregador.py, rotulos.py, routerbench.py
│   ├── avaliacao/            politicas.py, curva.py, relatorio.py
│   ├── ml/                   embedder.py, classificadores.py, roteador_v2.py
│   └── cli.py
├── dados/                    dataset.csv e arquivos de entrada
├── resultados/               registro.csv, ml_ocorrencias.csv, curva.csv
├── tests/                    espelha src/, um teste por módulo
└── README.md
```

**Regra de dependência:** `tipos` e `config` → (`tokenizador`, `adapters`, `detector`, `registro`, `limite`, `dados`) → (`roteador_v1`, `ml`) → `orquestrador` → `avaliacao` → `cli`. Nenhum módulo importa de um nível à direita e não há dependência circular. O orquestrador depende da interface `Roteador`, não de uma versão específica.

## Requisitos e instalação

- Python 3.11 ou superior
- Ollama em execução local (para o modelo barato e os embeddings)
- Chave da API do Gemini

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dependências de execução: `requests`, `numpy`, `scikit-learn`, `joblib`. Apenas para desenvolvimento: `pytest`.

## Configuração

Os valores ajustáveis ficam em `src/roteador/config.py`. Um JSON opcional pode sobrescrever os padrões:

```python
from roteador.config import carregar_config

cfg = carregar_config("config.json")
```

Chaves desconhecidas no JSON geram erro com mensagem clara. Os principais campos:

| Campo | Padrão | Descrição |
| --- | --- | --- |
| `modelo_barato` | `llama3.2:3b` | Modelo local via Ollama |
| `janela_contexto_barato` | `8192` | Contexto máximo aceito pelo barato |
| `limiar_tokens_pergunta` | `200` | Limiar do roteador v1 (a calibrar) |
| `x_tokens_forte_limitado` | `None` | Limite `X` para o forte; `None` desativa |
| `fator_teto` | `1.5` | Teto rígido = `ceil(fator_teto × X)` |
| `fator_nova_tentativa` | `2.0` | Multiplicador do teto na nova tentativa |
| `limiar_raciocinio` | `0.9` | Fração do teto consumida por raciocínio |
| `limiar_alerta` | `0.05` | Taxa de alerta no relatório |
| `percentis_faixas` | `(1/3, 2/3)` | Cortes das faixas de tamanho |
| `limiar_probabilidade` | `0.5` | Limiar do classificador A (v2) |
| `embedder_modelo` | `nomic-embed-text` | Modelo de embedding via Ollama |
| `preco_ref` | barato `(0.10, 0.50)`; forte `(2.00, 10.00)` | USD por milhão de tokens (entrada, saída) |

### Variáveis de ambiente

| Variável | Uso |
| --- | --- |
| `GEMINI_API_KEY` | Chave da API do Gemini (obrigatória para o adapter Gemini) |
| `GEMINI_MODEL` | Id do modelo Gemini usado como forte |
| `OLLAMA_HOST` | Endereço do Ollama (padrão `http://localhost:11434`) |

> Os preços em `preco_ref` são apenas **referências** para estimar custo (tabela da API Claude, outubro de 2026). Os modelos realmente usados neste projeto são gratuitos.

## Uso

> A interface abaixo corresponde ao plano de implementação (camadas 0 a 15). Subcomandos e opções são ativados conforme cada camada fica pronta.

### Rotear uma pergunta

```bash
roteador rotear "Explique a diferença entre treino e teste" --contexto documentos/guia.md
```

Mostra a decisão do roteador, a resposta e o custo estimado. O `--contexto` é opcional.

### Coletar respostas sobre o dataset

```bash
roteador coletar --politica v1
```

Políticas disponíveis: `barato`, `forte`, `forte_limitado`, `v1` (e `v2` após a camada 13). O comando roda o orquestrador sobre as perguntas do dataset e grava respostas, tokens e tempo, deixando as notas em branco para preenchimento manual.

### Avaliar e gerar a curva

```bash
roteador avaliar
```

Gera `resultados/curva.csv` e o relatório de alertas, sem chamar modelos.

### Treinar o roteador v2 (opcional)

```bash
roteador treinar
```

Treina os classificadores A e B e salva o modelo com `joblib`.

## Dados

O `dataset.csv` segue o esquema:

```
id,documento,tipo,pergunta,contexto_tokens,nota_barato,nota_forte,tokens_resposta_barato,tokens_resposta_forte,truncada_forte,nota_forte_limitado,tokens_resposta_forte_limitado
```

- Notas: `correta` (2), `parcial` (1) ou `errada` (0).
- `contexto_tokens`, `truncada_forte` e as duas últimas colunas são opcionais.
- **Barato basta** quando a nota do barato é maior ou igual à do forte.
- A divisão treino/teste é feita **por documento**, para o mesmo texto não aparecer nos dois lados.
- Resposta truncada sobe uma faixa de tamanho, pois o tamanho real é maior ou igual ao teto.

## Registros

### `resultados/registro.csv`

Uma linha por requisição:

```
id_requisicao,timestamp,papel,modelo,tokens_pergunta,tokens_entrada,tokens_saida,tokens_raciocinio,tempo_ms,custo_estimado_usd,motivo_decisao,faixa_prevista,limite_tokens,motivo_parada,ocorrencia,tentativas
```

### `resultados/ml_ocorrencias.csv`

Uma linha por ocorrência de resposta vazia ou truncada:

```
id_requisicao,timestamp,modelo,faixa_prevista,limite_aplicado,tokens_saida_reais,tokens_raciocinio,motivo_parada,tipo_ocorrencia,acao,resultado_final
```

Ações: `nenhuma`, `nova_tentativa_teto_maior`, `escalou_ao_forte`. Resultado final: `ok` ou `ainda_com_ocorrencia`.

## Curva de custo × qualidade

A avaliação compara estas políticas sobre o dataset já gravado (sem chamar modelos):

- `sempre_barato`
- `sempre_forte`
- `aleatorio(proporcao, semente)`
- `perfeito` (usa os rótulos reais; mostra o teto possível)
- `v1`
- `forte_limitado`
- `v2` (após a camada 13)

Para cada política são calculados a qualidade média (nota ÷ 2), o custo estimado e os percentuais em relação ao `sempre_forte`. O resultado principal é a frase:

> "Mantive **X%** da qualidade do modelo forte gastando **Y%** menos."

O relatório de alertas calcula a taxa de respostas vazias e truncadas por modelo e por faixa, comparando as ocorrências do arquivo de ML com as requisições do registro, e marca as que ultrapassam `limiar_alerta`.

## Testes

```bash
pytest
```

Nenhum teste usa rede, Ollama ou Gemini: os adapters falsos e as funções HTTP injetadas cobrem todo o fluxo.

## Limitações

- Apenas dois modelos participam do roteamento. Claude e DeepSeek existem como adapters opcionais, mas não entram na decisão desta versão.
- O custo é **estimado** por um preço de referência, já que os modelos usados são gratuitos.
- O roteador v1 não entende o conteúdo da pergunta: uma pergunta curta e pesada ("crie um CRUD completo") pode cair no barato.
- O v2 depende de dados rotulados suficientes e é avaliado por documento para evitar vazamento.
- Não há treinamento ou ajuste fino de LLM, interface web nem autenticação.
- Dataset e par de modelos do RouterBench ainda serão definidos.

