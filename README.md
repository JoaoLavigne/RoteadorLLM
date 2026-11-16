# Roteador de LLMs (custo × qualidade)

> Um roteador que escolhe, para cada pergunta, o modelo mais barato que ainda entrega a qualidade necessária.

O projeto compara um modelo **barato** (`llama3.2:3b`, local via Ollama) com um modelo **forte**. Cada requisição é decidida, registrada e avaliada em uma curva de custo × qualidade.

O resultado principal é uma frase simples:

> “Mantive **X%** da qualidade do modelo forte gastando **Y%** menos.”

## Por que este projeto existe

Produtos com LLM pagam por token. Modelos fortes custam muito mais que modelos pequenos. Ao mesmo tempo, a maior parte das perguntas é simples. Enviar tudo ao modelo forte desperdiça dinheiro. Enviar tudo ao modelo pequeno derruba a qualidade nas perguntas difíceis.

Este projeto testa uma alternativa: **decidir por requisição**. O roteador escolhe o modelo, aplica limites quando faz sentido, registra a decisão e mede o custo e a qualidade.

## O que está pronto

- **Roteador v1:** decide por contagem de tokens e pela janela de contexto.
- **Roteador v2:** usa dois classificadores de ML sobre o embedding da pergunta.
- **Limite de tokens:** o modelo forte pode receber um teto para respostas de tamanho médio.
- **Retry e fallback:** falhas transitórias são repetidas; se necessário, o outro modelo é tentado.
- **Detector de respostas vazias ou truncadas:** classifica a resposta e faz no máximo uma nova tentativa.
- **Registro:** grava modelo, tokens, tempo, custo estimado e decisão.
- **Dataset:** lê CSV ou SQLite e separa treino e teste por documento.
- **Avaliação:** gera a curva de custo × qualidade e o relatório de alertas.
- **Testes:** a suíte cobre o fluxo sem rede, Ollama ou Gemini.

## Como funciona

```text
pergunta + contexto
        │
        ▼
   tokenizador
        │
        ▼
  roteador v1 ou v2
        │
        ├──────────────► modelo barato (Ollama)
        │
        └──────────────► modelo forte 
        │
        ▼
  orquestrador + adapter
        │
        ▼
  detector de falhas
        │
        ▼
  resposta + registro + arquivo de ML
```

1. O tokenizador conta os tokens da pergunta e do contexto.
2. O roteador devolve uma `Decisao`: papel, motivo, faixa prevista e limite de tokens.
3. O orquestrador chama o adapter do modelo escolhido.
4. O detector classifica a resposta como completa, vazia ou truncada.
5. Respostas vazias ou truncadas recebem uma única nova tentativa.
6. O custo é estimado e a requisição é registrada.
7. A avaliação compara políticas sobre o dataset já gravado.

### v1 e v2

| Versão | Como decide | Quando usar |
| --- | --- | --- |
| **v1** | Contagem de tokens da pergunta e janela de contexto | Primeiro uso, sem modelo treinado |
| **v2** | Classificador A (“o barato basta?”) + classificador B (faixa da resposta) | Depois de treinar com dados rotulados(Router Bench) |

O v2 usa o embedding da pergunta e o `log1p` da contagem de tokens. O modelo B prevê a faixa `curta`, `media` ou `longa`. No forte com faixa média, o roteador aplica um limite de tokens.

### Resultados medidos

Teste feito no recorte do RouterBench com 6.000 prompts, usando o mesmo conjunto de teste de 1.200 linhas e o par Yi-34B-Chat (barato) / gpt-4 (forte):

| Versão | Configuração | Qualidade vs. forte | Economia |
| --- | --- | ---: | ---: |
| **v1** | limiar de 100 tokens | 94,4% | 14,2% |
| **v2** | limiar de probabilidade 0,81 | 97,0% | 17,9% |


## Requisitos

- Python 3.11 ou superior
- [Ollama](https://ollama.com/) em execução local
- Chave da API da LLM

O Ollama é usado pelo modelo barato e pelos embeddings do v2. 

## Instalação

```bash
git clone <url-do-repositorio>
cd RoteadorLLM

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Baixe os modelos locais:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Configure a chave da API de sua LLM de uso

Variáveis disponíveis:

| Variável | Uso |
| --- | --- |
| `OLLAMA_HOST` | Endereço do Ollama (padrão `http://localhost:11434`) |

## Configuração

Os valores padrão ficam em `src/roteador/config.py`. Você pode sobrescrever os valores com um arquivo JSON:

```json
{
  "modelo_barato": "llama3.2:3b",
  "tokenizador": "tiktoken:o200k_base",
  "limiar_tokens_pergunta": 200,
  "limiar_probabilidade": 0.5
}
```

Use o arquivo em qualquer subcomando:

```bash
roteador rotear "Explique treino e teste" --config config.json
```

Chaves desconhecidas geram erro com mensagem clara. O arquivo `config.json` é local e não entra no Git.

## Início rápido

Com o Ollama e o Gemini configurados:

```bash
# 1. Roteie uma pergunta
roteador rotear "Explique a diferença entre treino e teste"

# 2. Colete respostas sobre o dataset
roteador coletar --politica v1

# 3. Gere a curva de custo × qualidade
roteador avaliar
```

O subcomando `rotear` usa o **v1**. Para usar o v2, treine o modelo e use `coletar --politica v2` ou `avaliar` com o modelo disponível.

### Treinar o v2

```bash
roteador treinar
```

O comando divide o dataset por documento, treina os classificadores A e B, mostra a acurácia no conjunto de teste e salva o modelo em `resultados/modelo_v2.joblib`.

### Calibrar o limiar de probabilidade do v2

```bash
roteador treinar-limiar-probabilidade \
  --routerbench dados/routerbench/recorte.csv \
  --barato zero-one-ai/Yi-34B-Chat \
  --forte gpt-4-1106-preview \
  --meta 0.95
```

O comando converte o par de modelos do RouterBench, divide por prompt, treina A e B, calibra `limiar_probabilidade` na validação, confirma no teste e grava o modelo, `calibracao_limiar_probabilidade.csv` e `config_limiar_probabilidade.json`. Se a meta falhar, a configuração não é adotada e o código de saída é 1.

O recorte versionado é `dados/routerbench/recorte_yi_gpt4_0shot.csv` (6.000 prompts do par Yi-34B/gpt-4). A calibração de referência usa `--meta 0.95` e o embedder `nomic-embed-text`.

Essa etapa gera uma pré-calibração. A validação final com dados próprios é a Implementação 17 do SPEC e exige um `dados/dataset.csv` real rotulado.

## Comandos

| Comando | O que faz |
| --- | --- |
| `roteador rotear "pergunta"` | Responde uma pergunta com o roteador v1. Aceita `--contexto arquivo`. |
| `roteador coletar --politica v1` | Roda uma política sobre o dataset e grava respostas para anotação. |
| `roteador avaliar` | Gera `curva.csv` e o relatório de alertas sem chamar os modelos de resposta. |
| `roteador treinar` | Treina e salva o modelo v2. |
| `roteador treinar-limiar-tokens` | Calibra o limiar de tokens do v1 na validação e grava a configuração. |
| `roteador treinar-limiar-probabilidade` | Treina o v2 com o RouterBench, calibra `limiar_probabilidade` e grava modelo, CSV e configuração. |

Políticas de coleta: `barato`, `forte`, `forte_limitado`, `v1` e `v2`.

Todos os comandos aceitam `--config arquivo.json`.

## Dataset

O arquivo padrão é `dados/dataset.csv`:

```text
id,documento,tipo,pergunta,contexto_tokens,nota_barato,nota_forte,tokens_resposta_barato,tokens_resposta_forte,truncada_forte,nota_forte_limitado,tokens_resposta_forte_limitado
```

Regras principais:

- Notas: `correta` (2), `parcial` (1) ou `errada` (0).
- “O barato basta” quando `nota_barato >= nota_forte`.
- A separação treino/teste é feita **por documento**.
- Resposta truncada sobe uma faixa de tamanho.
- O mesmo esquema pode ficar em SQLite (`.db`, `.sqlite` ou `.sqlite3`).

A calibração do `limiar_tokens_pergunta` exige dataset real rotulado: recomendado de 300 a 500 perguntas em 30 a 50 documentos, divididos por documento em treino, validação e teste. Dados sintéticos servem apenas para os testes automatizados.

## Arquitetura

```text
.
├── pyproject.toml
├── README.md
├── SPEC.md
├── Roteador de requisições para LLMs (custo × qualidade).md
├── src/roteador/
│   ├── tipos.py              tipos, enums e interfaces
│   ├── config.py             valores ajustáveis
│   ├── tokenizador.py        contagem de tokens
│   ├── adapters/             base, falso, Ollama e Gemini
│   ├── detector.py           respostas vazias ou truncadas
│   ├── comunicacao.py        retry e fallback
│   ├── registro.py           CSV e memória
│   ├── limite.py             limite e teto de tokens
│   ├── roteador_v1.py        decisão por tokens
│   ├── orquestrador.py       fluxo completo
│   ├── dados/                carregador e rótulos (RouterBench futuro)
│   ├── avaliacao/            políticas, curva e relatório
│   ├── ml/                   embedder, classificadores e roteador_v2
│   └── cli.py
├── dados/                    dataset e entradas
├── resultados/               arquivos gerados
└── tests/                    testes sem rede
```

O orquestrador depende das interfaces `Roteador`, `Adapter` e `Registro`. Ele não depende de uma versão específica do roteador nem do formato CSV.

## Testes

```bash
pytest
```

Os testes não usam rede, Ollama ou Gemini. Eles usam adapters falsos, `tmp_path` e funções HTTP injetadas.

### Teste de economia

```bash
pytest tests/avaliacao/test_politicas.py::test_avalia_politicas_com_resultado_calculado_a_mao -v
```


### Ganho do limiar calibrado (cenário simulado)

> **Cenário simulado (temporário).** Este número valida a mecânica do `ml/limiar.py`; não é um benchmark real nem vai ser usado como resultado do produto.

```bash
pytest tests/ml/test_limiar.py::test_calibrar_escolhe_limiar_com_economia_no_cenario_simulado -v
```

Cenário: 10 documentos com 2 perguntas cada. Nas perguntas curtas (10 tokens), o barato acerta; nas longas (100 tokens), o barato erra e o forte acerta. Preços padrão de `config.py`. A calibração divide por documento em treino, validação e teste, escolhe o maior limiar que mantém a meta na validação e confirma no teste.

| Limiar escolhido | Qualidade vs. forte no teste | Economia no teste | Meta |
| ---: | ---: | ---: | ---: |
| 82 tokens | 100,0% | 45,7% | 95,0% |

Em resumo: **no cenário simulado, manteve 100% da qualidade do forte gastando 45,7% menos**. O dataset real ainda precisa ser coletado e rotulado.

#### Glossário das políticas

| Política | O que é e o que faz |
| --- | --- |
| `sempre_barato` | Manda todas as linhas para o modelo barato (`llama3.2:3b`) e usa a nota e os tokens de resposta do barato. |
| `sempre_forte` | Manda todas as linhas para o modelo forte. É a referência de 100% de qualidade e de custo. |
| `aleatorio` | Escolhe o barato em 50% das linhas, com semente fixa (`42`), e o forte no restante. |
| `perfeito` | Usa os rótulos reais como oráculo: escolhe o barato quando `nota_barato >= nota_forte` e o forte caso contrário. Mostra o teto possível, mas não é implementável em produção. |
| `v1` | Decide por contagem de tokens: vai ao barato se a pergunta couber no `limiar_tokens_pergunta` e a pergunta + contexto couber na `janela_contexto_barato`; caso contrário, vai ao forte. |
| `v2` | Usa o classificador A (probabilidade de “o barato basta”) e o classificador B (faixa da resposta). Vai ao barato quando a probabilidade supera `limiar_probabilidade`; no forte com faixa média, aplica limite de tokens. |
| `forte_limitado` | Manda todas as linhas ao forte, mas usa `nota_forte_limitado` e `tokens_resposta_forte_limitado`, isto é, o resultado do forte com teto de tokens. |

## Limitações

- Apenas dois modelos participam da decisão.
- O v1 não entende o conteúdo da pergunta. Uma pergunta curta e pesada pode cair no barato.
- O v2 precisa de dados rotulados suficientes.
- O custo é estimado por uma tabela de referência.
- Não há interface web, autenticação ou ajuste fino de LLM.


## Próximos passos

- Calibrar `limiar_tokens_pergunta` com o dataset real (comando implementado).
- Calibrar `limiar_probabilidade` com o RouterBench.
- Criar um servidor HTTP local compatível com OpenAI.
- Ampliar o dataset e validar em mais documentos.

## Documentação

- [PRD](<Roteador de requisições para LLMs (custo × qualidade).md>) — problema, objetivos, requisitos e métricas.
- [SPEC](SPEC.md) — arquitetura detalhada e implementações.
- [Relatório de calibração do tokenizador](relatório_calibração_tokenizador.md).
