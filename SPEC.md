# Guia de implementação do roteador de LLMs

*Instruções para a LLM que vai programar o projeto, uma camada por vez. Base: documento “Roteador de requisições para LLMs (custo × qualidade)”.*

## 1. Como usar este documento

1. A cada pedido, cole as seções 2 a 6, a seção da camada pedida (seção 7) e um resumo das camadas já prontas: só nomes de arquivos e assinaturas de funções.
2. Peça **uma camada por vez**. Cada camada termina com testes passando.
3. Ao terminar uma camada, atualize a seção 8.

**Regras para a LLM**

- Implemente só a camada pedida. Se faltar informação, pergunte em vez de inventar.
- Cada camada depende apenas das anteriores (seção 4.2), sem dependência circular.
- Nenhum teste usa rede, Ollama ou Gemini: use o adapter falso ou funções HTTP injetadas.
- Valores marcados como PROVISÓRIO ficam em `config.py`, nunca dentro da lógica.
- Funções pequenas e puras; I/O só nas bordas (adapters, registro, dados, CLI).
- Dependências permitidas: `requests`, `numpy`, `scikit-learn`, `joblib` e, só para desenvolvimento, `pytest`. CSV com o módulo `csv` da biblioteca padrão. Qualquer outra, só com autorização.
- Nomes de código e comentários em português. Cada arquivo novo vem com o teste correspondente em `tests/`.

## 2. Contexto e premissas

O roteador decide, para cada pergunta, se responde o modelo barato (local) ou o forte (hospedado), registra cada chamada e é avaliado por uma curva de custo × qualidade.

- **Barato:** `llama3.2:3b` via Ollama. **Forte:** Gemini Flash.
- **v1:** decide só pela contagem de tokens da pergunta (sem lista de palavras) e por uma regra de janela de contexto.
- **v2:** dois classificadores sobre o embedding da pergunta e a contagem de tokens. O A responde “o barato basta?” e o B prevê a faixa de tamanho da resposta (curta, média ou longa).
- **Limite de tokens:** respostas de faixa média vão ao forte com o prompt “Limite seus gastos a cerca de X tokens” e um teto rígido maior que X.
- **Respostas vazias ou truncadas:** são detectadas, recebem uma nova tentativa e são gravadas no arquivo de ML.
- **Premissa a confirmar:** Python 3.11 ou superior, pacote `roteador` em layout `src/`, testes com `pytest`. Se o projeto usar outra linguagem, só esta premissa e os nomes de arquivo mudam.

## 3. Decisões e valores provisórios

| Ponto | Decisão | Onde entra |
| --- | --- | --- |
| Modelo forte | Gemini Flash. O id exato vem da variável de ambiente `GEMINI_MODEL` e a chave de `GEMINI_API_KEY`. DeepSeek fica fora desta versão. | `adapters/gemini.py` |
| Preço de referência | Preços da API Claude em USD por milhão de tokens (entrada / saída): barato = Haiku 5.5 (0,10 / 0,50); forte = Sonnet 5.5 (2 / 10). Valores publicados em outubro de 2026; conferir na [página de preços](https://platform.claude.com/docs/en/about-claude/pricing) antes de publicar. Servem só para estimar custo, porque os modelos realmente usados são gratuitos. | `config.preco_ref` |
| Tamanho do dataset | Sem definição. O código não presume quantidade; a divisão por documento avisa quando há menos de 2 documentos. | camada 9 |
| Registro de requisições | CSV. | camada 5 |
| Cortes das faixas e valor de X | Definir depois. Provisório: terços das respostas reais, com X igual ao limite superior da faixa média. | `config.py`, camadas 9 e 13 |
| Fator do teto e limiar de alerta | Definir depois. Provisório: 1,5 × X e 5%. | `config.py` |
| Tokenizador | Definir depois. Provisório: caracteres ÷ 4, atrás da interface `Tokenizador`, para trocar sem mexer no resto. | camada 1 |
| Meta de qualidade do limiar de tokens | Definir depois. Provisório: 95% da qualidade do forte. | camada 14 (opcional) |
| Par de modelos e meta de qualidade do RouterBench | Definir depois. | camada 15 (opcional) |
| Parâmetros e motivos de parada de cada provedor | Definir depois. Usar a seção 6 e conferir na documentação de cada provedor. | camada 3 |
| Nome do projeto | Sem definição. Nome de pacote provisório: `roteador`. | `pyproject.toml` |

## 4. Arquitetura

### 4.1 Pastas

```
roteador/
├── pyproject.toml
├── src/roteador/
│   ├── tipos.py              dataclasses, enums e interfaces (Protocol)
│   ├── config.py             todos os valores ajustáveis
│   ├── tokenizador.py
│   ├── adapters/             base.py, falso.py, ollama.py, gemini.py
│   ├── detector.py           classifica respostas vazias ou truncadas
│   ├── registro.py           CSV de requisições e arquivo de ML
│   ├── limite.py             prompt de limite e teto rígido
│   ├── roteador_v1.py
│   ├── orquestrador.py       junta tudo e trata a nova tentativa
│   ├── dados/                carregador.py, rotulos.py, routerbench.py
│   ├── avaliacao/            politicas.py, curva.py, relatorio.py
│   ├── ml/                   embedder.py, classificadores.py, roteador_v2.py, limiar.py
│   ├── servidor.py
│   └── cli.py
├── dados/                    dataset.csv e arquivos de entrada
├── resultados/               registro.csv, ml_ocorrencias.csv, curva.csv
└── tests/                    espelha src/, um teste por módulo
```

### 4.2 Regra de dependência

`tipos` e `config` → (`tokenizador`, `adapters`, `detector`, `registro`, `limite`, `dados`) → (`roteador_v1`, `ml`) → `orquestrador` → `avaliacao` → `cli`.

Um módulo nunca importa de um nível à direita. O orquestrador depende da interface `Roteador`, e não de uma versão específica.

### 4.3 Fluxo de uma requisição

1. O `Tokenizador` conta os tokens da pergunta.
2. O `Roteador` devolve uma `Decisao`: papel (barato ou forte), motivo, faixa prevista e limite de tokens.
3. Se há limite X (só no forte), `limite.montar_prompt` acrescenta “Limite seus gastos a cerca de X tokens.” e define o teto rígido.
4. O adapter do papel gera a resposta (`RespostaModelo`).
5. O `detector` classifica a resposta.
6. Se for vazia ou truncada, há **uma única** nova tentativa: o forte repete com o teto dobrado; o barato escala ao forte, sem limite.
7. O custo estimado é calculado e a requisição é gravada no registro; se houve ocorrência, ela também vai para o arquivo de ML.
8. O orquestrador devolve um `Resultado`.

### 4.4 Contratos (`tipos.py`)

```python
class Papel(str, Enum): BARATO = 'barato'; FORTE = 'forte'
class Faixa(str, Enum): CURTA = 'curta'; MEDIA = 'media'; LONGA = 'longa'
class MotivoParada(str, Enum): COMPLETO = 'completo'; TRUNCADO = 'truncado'; VAZIO = 'vazio'
class TipoOcorrencia(str, Enum):
    NENHUMA = 'nenhuma'; VAZIO = 'vazio'; TRUNCADO = 'truncado'
    POSSIVELMENTE_TRUNCADO = 'possivelmente_truncado'
    LIMITE_CONSUMIDO_PELO_RACIOCINIO = 'limite_consumido_pelo_raciocinio'

@dataclass(frozen=True)
class RespostaModelo:
    texto: str; tokens_entrada: int; tokens_saida: int
    tokens_raciocinio: int | None; motivo_parada: MotivoParada
    tempo_ms: float; modelo: str

@dataclass(frozen=True)
class Decisao:
    papel: Papel; motivo: str
    faixa_prevista: Faixa | None = None; limite_tokens: int | None = None

@dataclass(frozen=True)
class Ocorrencia: tipo: TipoOcorrencia; alerta: bool; motivo: str

@dataclass(frozen=True)
class Resultado:
    resposta: RespostaModelo; decisao: Decisao; ocorrencia: Ocorrencia
    custo_estimado_usd: float; tentativas: int

class Tokenizador(Protocol):
    def contar(self, texto: str) -> int: ...
class Adapter(Protocol):
    nome: str
    def gerar(self, prompt: str, limite_tokens: int | None) -> RespostaModelo: ...
class Roteador(Protocol):
    def decidir(self, pergunta: str, contexto: str = '') -> Decisao: ...
```

### 4.5 Configuração (`config.py`)

```python
@dataclass
class Config:
    modelo_barato: str = 'llama3.2:3b'
    janela_contexto_barato: int = 8192            # PROVISÓRIO
    limiar_tokens_pergunta: int = 200             # PROVISÓRIO: calibrar na curva
    x_tokens_forte_limitado: int | None = None    # X; PROVISÓRIO (None = sem limite)
    fator_teto: float = 1.5                       # PROVISÓRIO
    fator_nova_tentativa: float = 2.0             # PROVISÓRIO
    limiar_raciocinio: float = 0.9                # PROVISÓRIO
    limiar_alerta: float = 0.05                   # PROVISÓRIO
    percentis_faixas: tuple = (1/3, 2/3)          # PROVISÓRIO
    limiar_probabilidade: float = 0.5             # PROVISÓRIO (v2)
    embedder_modelo: str = 'nomic-embed-text'     # PROVISÓRIO
    preco_ref: dict = field(default_factory=lambda: {'barato': (0.10, 0.50), 'forte': (2.00, 10.00)})  # USD por milhão de tokens (entrada, saída)
    caminho_registro: str = 'resultados/registro.csv'
    caminho_ml: str = 'resultados/ml_ocorrencias.csv'
    caminho_dataset: str = 'dados/dataset.csv'
```

## 5. Esquemas dos arquivos CSV

Primeira linha de cada arquivo (cabeçalho):

```
registro.csv
id_requisicao,timestamp,papel,modelo,tokens_pergunta,tokens_entrada,tokens_saida,tokens_raciocinio,tempo_ms,custo_estimado_usd,motivo_decisao,faixa_prevista,limite_tokens,motivo_parada,ocorrencia,tentativas

ml_ocorrencias.csv
id_requisicao,timestamp,modelo,faixa_prevista,limite_aplicado,tokens_saida_reais,tokens_raciocinio,motivo_parada,tipo_ocorrencia,acao,resultado_final

dataset.csv
id,documento,tipo,pergunta,contexto_tokens,nota_barato,nota_forte,tokens_resposta_barato,tokens_resposta_forte,truncada_forte,nota_forte_limitado,tokens_resposta_forte_limitado
```

No `dataset.csv`, as notas são `correta`, `parcial` ou `errada`; `contexto_tokens`, `truncada_forte` e as duas últimas colunas são opcionais. Os valores de `acao` são `nenhuma`, `nova_tentativa_teto_maior` e `escalou_ao_forte`; os de `resultado_final`, `ok` e `ainda_com_ocorrencia`.

## 6. Referências dos provedores (a confirmar na documentação)

| Provedor | Parâmetro de limite | Motivo de parada por limite | Tokens |
| --- | --- | --- | --- |
| Ollama (`POST /api/chat`, `stream: false`) | `options.num_predict` | `done_reason: length` | `prompt_eval_count`, `eval_count` |
| Gemini (`generateContent`) | `generationConfig.maxOutputTokens` | `finishReason: MAX_TOKENS` | `usageMetadata.promptTokenCount`, `candidatesTokenCount`, `thoughtsTokenCount` |
| Claude | `max_tokens` | `stop_reason: max_tokens` | `usage.input_tokens`, `usage.output_tokens` |
| DeepSeek | `max_tokens` | `finish_reason: length` | `usage.prompt_tokens`, `usage.completion_tokens` |

Qualquer ajuste de nome de campo é feito no adapter, nunca na lógica do roteador.

## 7. Camadas de implementação

As camadas 0 a 11 formam o **plano mínimo** (roteador v1 completo, com avaliação e CLI). As camadas 12 e 13 formam o **plano completo** (v2). As camadas 14 a 16 são opcionais.

### Camada 0: esqueleto, tipos e configuração

- **Arquivos:** `pyproject.toml`, `tipos.py`, `config.py`, `tests/test_config.py`.
- **Fazer:** criar o pacote, os tipos da seção 4.4 e a `Config` da seção 4.5. `carregar_config(caminho=None)` lê um JSON opcional que sobrescreve os padrões e falha com mensagem clara se houver chave desconhecida.
- **Testes:** padrões corretos; o JSON sobrescreve; chave desconhecida falha.
- **Pronto quando:** `pytest` roda e passa.

### Camada 1: tokenizador real


1. **Escolher uma biblioteca e usar sempre a mesma.** Para o roteador, a consistência importa mais que a exatidão. `tiktoken` com uma codificação fixa em `cfg.tokenizador` (PROVISÓRIO, por exemplo `tiktoken:o200k_base`). A alternativa é a biblioteca `tokenizers` com o arquivo de tokenizador do Llama 3.2, que dá a contagem exata do modelo barato, mas exige baixar o arquivo. Esta camada autoriza a nova dependência.
2. **Medir o erro do aproximado.** Com as 50 perguntas do dataset, compare a contagem do aproximado com a do real e calcule a razão média. Em português, o divisor 4 tende a subestimar os tokens, então ajuste o divisor se a diferença for grande e registre o valor em `config.py`.
3. **Implementar `TokenizadorReal`.** Carrega a biblioteca só no primeiro uso (carga preguiçosa) e levanta `ErroTokenizador` se a biblioteca não estiver instalada ou o arquivo da codificação não puder ser obtido.
4. **Ligar na fábrica.** `criar_tokenizador(cfg)` passa a montar `TokenizadorComReserva(TokenizadorReal(...), TokenizadorAproximado())`.
5. **Registrar qual tokenizador contou.** Acrescente a coluna `tokenizador_usado` (`real` ou `aproximado`) ao `registro.csv`, para não misturar contagens diferentes no treino do classificador.
6. **Testes.** `TokenizadorReal` com `pytest.importorskip` (é pulado se a biblioteca não existir); mesmo texto, mesma contagem; texto vazio dá 0; a contagem real fica dentro de uma tolerância esperada em relação ao aproximado; falha simulada na carga aciona a reserva.
7. **Pronto quando:** o roteador v1 roda com o real, cai para o aproximado se a carga falhar, e o registro mostra qual dos dois foi usado.

### Camada 1.2 Tokenizador substituto
- **Arquivo:** `tokenizador.py`.
- **Fazer:** `TokenizadorAproximado`, que devolve `ceil(len(texto) / 4)` e 0 para texto vazio.
- **Testes:** texto vazio, curto, longo e com acentos; o objeto satisfaz o `Protocol`.
- **Não fazer:** instalar biblioteca de tokenização (a escolha foi adiada).


### Camada 2: contrato de adapter e adapter falso

- **Arquivos:** `adapters/base.py`, `adapters/falso.py`.
- **Fazer:** `AdapterFalso(respostas)` devolve as respostas programadas em sequência e guarda cada chamada recebida (prompt e limite), para os testes inspecionarem.
- **Testes:** a sequência é respeitada; as chamadas ficam registradas; esgotar a lista levanta erro claro.
- **Pronto quando:** o orquestrador pode ser testado sem rede.

### Camada 3: adapters Ollama e Gemini

- **Arquivos:** `adapters/ollama.py`, `adapters/gemini.py`.
- **Fazer:** cada adapter recebe uma função `post` injetável (padrão `requests.post`), converte a resposta do provedor em `RespostaModelo` (seção 6), normaliza o motivo de parada e mede o tempo. Campo ausente na resposta nunca gera exceção: vira texto vazio e motivo `vazio`. Erro de rede ou HTTP vira `ErroAdapter` com mensagem clara. O Gemini falha com mensagem clara se `GEMINI_API_KEY` ou `GEMINI_MODEL` faltarem.
- **Testes** (com `post` falso e JSONs de exemplo): resposta normal; parada por limite; resposta do Gemini sem `parts` (o raciocínio consumiu o limite) vira `vazio`; tokens de raciocínio lidos quando existirem; erro HTTP.
- **Pronto quando:** os testes passam sem rede.

### Camada 4: detector de respostas vazias ou truncadas

- **Arquivo:** `detector.py`.
- **Fazer:** `classificar(resposta, limite_tokens, cfg) -> Ocorrencia`, aplicando as regras **nesta ordem**:
  1. texto vazio com tokens de saída > 0: `VAZIO`, alerta de provável gasto em raciocínio;
  2. tokens de saída = 0, sem erro: `VAZIO`, alerta;
  3. tokens de raciocínio ≥ `limiar_raciocinio` × teto: `LIMITE_CONSUMIDO_PELO_RACIOCINIO`, alerta;
  4. motivo de parada `truncado`: `TRUNCADO`, alerta;
  5. saída ≥ 95% do teto e texto sem pontuação final: `POSSIVELMENTE_TRUNCADO`;
  6. caso contrário, `NENHUMA`.
- **Testes:** um para cada regra; a ordem das regras; resposta normal; sem limite (`None`), as regras 3 e 5 não se aplicam.
- **Pronto quando:** todas as regras estão cobertas.

### Camada 5: registro de requisições e arquivo de ML

- **Arquivo:** `registro.py`.
- **Fazer:** `RegistroCsv(caminho, colunas)` cria o arquivo com cabeçalho se não existir e `adicionar(linha)` grava uma linha e fecha o arquivo a cada escrita (nada se perde se a execução for interrompida). Duas instâncias usam os esquemas da seção 5. Inclua a função de leitura.
- **Testes** (com `tmp_path`): cabeçalho criado uma vez; várias linhas sem duplicar o cabeçalho; campo faltando ou sobrando gera erro; a leitura devolve o que foi gravado, inclusive com vírgulas, aspas e acentos no texto.

### Camada 6: limite de tokens

- **Arquivo:** `limite.py`.
- **Fazer:** `montar_prompt(prompt, x)` acrescenta ao final “Limite seus gastos a cerca de X tokens.”; `teto_rigido(x, cfg)` devolve `ceil(fator_teto × x)`; `limite_por_faixa(faixa, cortes)` devolve X (o corte superior da faixa média) para a faixa média e `None` para as outras.
- **Testes:** texto exato do prompt; arredondamento do teto; faixas curta e longa sem limite; X ≤ 0 gera erro.

### Camada 7: roteador v1

- **Arquivo:** `roteador_v1.py`.
- **Fazer:** `RoteadorV1(tokenizador, cfg).decidir(pergunta, contexto)`:
  1. se tokens da pergunta + tokens do contexto > `janela_contexto_barato`, vai ao forte (“contexto não cabe no barato”);
  2. se tokens da pergunta > `limiar_tokens_pergunta`, vai ao forte (“pergunta acima do limiar”);
  3. caso contrário, vai ao barato. No forte, devolve `limite_tokens = cfg.x_tokens_forte_limitado`. A faixa prevista é `None`.
- **Testes:** abaixo, igual e acima do limiar; contexto grande; motivo preenchido; duas perguntas com a mesma contagem de tokens e textos diferentes recebem a mesma decisão.
- **Não fazer:** lista de palavras-chave.

### Camada 8: orquestrador

- **Arquivo:** `orquestrador.py`.
- **Fazer:** `Orquestrador(roteador, adapters, tokenizador, registro, registro_ml, cfg).responder(pergunta, contexto='')` segue o fluxo da seção 4.3. A nova tentativa acontece no máximo uma vez, para `VAZIO`, `TRUNCADO` e `LIMITE_CONSUMIDO_PELO_RACIOCINIO`; `POSSIVELMENTE_TRUNCADO` só gera alerta. Custo estimado = tokens × preço de referência do papel, separando entrada e saída. Cada requisição vai ao registro, e cada ocorrência ao arquivo de ML.
- **Testes** (com adapters falsos): fluxo normal; barato vazio escala ao forte; forte truncado repete com o teto dobrado; segunda tentativa também falha e o resultado volta com a ocorrência, sem terceira tentativa; X aplicado só ao forte (conferir o prompt recebido); custo calculado; linhas gravadas nos dois arquivos.
- **Pronto quando:** todo o fluxo roda sem rede.

### Camada 9: dados e rótulos

- **Arquivos:** `dados/carregador.py`, `dados/rotulos.py`.
- **Fazer:**
  - ler o `dataset.csv` (seção 5), com erro que cita o número da linha para nota inválida;
  - `barato_basta(linha)`: verdadeiro quando a nota do barato é maior ou igual à do forte (correta = 2, parcial = 1, errada = 0);
  - `calcular_cortes(tokens, percentis)` devolve os dois cortes das faixas e `faixa_de(tokens, cortes)` devolve a faixa;
  - resposta truncada (`truncada_forte` verdadeiro) sobe uma faixa, porque o tamanho real é maior ou igual ao teto (longa continua longa);
  - `dividir_por_documento(linhas, fracao_teste, semente)` nunca separa linhas do mesmo documento e falha com mensagem clara se houver menos de 2 documentos.
- **Testes:** todas as combinações de notas; cortes com poucos dados; fronteiras das faixas; truncada sobe uma faixa; divisão sem vazamento entre treino e teste; erro com 1 documento.

### Camada 10: avaliação e curva custo × qualidade

- **Arquivos:** `avaliacao/politicas.py`, `curva.py`, `relatorio.py`.
- **Fazer:** avaliar offline, sobre o dataset já gravado e sem chamar modelos, estas políticas: `sempre_barato`, `sempre_forte`, `aleatorio(proporcao, semente)`, `perfeito` (barato quando `barato_basta`), `v1` e `forte_limitado` (usa as colunas `nota_forte_limitado` e `tokens_resposta_forte_limitado`; sem elas, é ignorada com aviso). Para cada política: qualidade média (nota ÷ 2), custo estimado e os percentuais em relação ao `sempre_forte`. Gerar `curva.csv` e uma tabela em Markdown no formato “mantive X% da qualidade do forte gastando Y% menos”. O relatório de alertas calcula a taxa de respostas vazias e truncadas por modelo e por faixa (ocorrências do arquivo de ML ÷ requisições do registro) e marca as que passam de `limiar_alerta`.
- **Testes:** dataset pequeno com resultado calculado à mão; `perfeito` nunca perde em qualidade; `sempre_forte` dá 100% por definição; aleatório reprodutível com a mesma semente; taxa de alertas correta.
- **Pronto quando:** a curva sai de um dataset de exemplo.

### Camada 11: CLI e README

- **Arquivos:** `cli.py`, `README.md`.
- **Fazer:** subcomandos `rotear 'pergunta' [--contexto arquivo]`, `coletar --politica {barato,forte,forte_limitado,v1}` (roda o orquestrador sobre as perguntas do dataset e grava respostas, tokens e tempo, com as notas em branco para preenchimento manual) e `avaliar` (curva e relatório de alertas). O README explica instalação, variáveis de ambiente, cada comando, como ler a curva e os limites do projeto.
- **Testes:** cada subcomando com adapters falsos e `tmp_path`; mensagens de erro para chave ausente e dataset inexistente.
- **Pronto quando:** o plano mínimo roda de ponta a ponta.

### Camada 12: embedder e classificadores A e B

- **Arquivos:** `ml/embedder.py`, `ml/classificadores.py`.
- **Fazer:**
  - `Embedder` (Protocol) com `embutir(textos) -> np.ndarray`; `EmbedderOllama` (modelo em `cfg.embedder_modelo`, `post` injetável) e `EmbedderFalso` (vetores determinísticos a partir do hash do texto, para testes);
  - `montar_features(embeddings, tokens_pergunta)` junta o embedding e o `log1p` dos tokens;
  - `treinar_a` (regressão logística para `barato_basta`) e `treinar_b` (multiclasse para a faixa);
  - `avaliar` devolve acurácia, precisão e revocação por classe e a matriz de confusão;
  - `salvar` e `carregar` com `joblib`, guardando o nome do embedder e os cortes das faixas.
- **Testes** (com `EmbedderFalso` e dados sintéticos): o modelo aprende uma regra simples (acurácia acima de 0,9); formato das features; salvar e carregar dão a mesma predição; matriz de confusão com o tamanho certo; carregar um modelo treinado com outro embedder gera erro claro.

### Camada 13: roteador v2

- **Arquivo:** `ml/roteador_v2.py`.
- **Fazer:** `RoteadorV2(embedder, modelo_a, modelo_b, tokenizador, cfg).decidir(...)`: vai ao barato se a probabilidade de “o barato basta” for ≥ `limiar_probabilidade`; caso contrário, ao forte. A faixa prevista vem do B. No forte, com faixa média, `limite_tokens = limite_por_faixa(...)`. O motivo registra a probabilidade. Adicionar o subcomando `treinar` e a política `v2` na CLI e na curva.
- **Testes:** modelos falsos com probabilidade conhecida; limiar nas bordas; faixa média no forte gera limite, curta e longa não; o barato nunca recebe limite; satisfaz o `Protocol` `Roteador`.

### Camada 14: treinar `limiar_tokens_pergunta` na curva

- **Objetivo:** achar o valor de `limiar_tokens_pergunta` que dá o melhor equilíbrio custo × qualidade no roteador v1. A calibração usa os dados rotulados próprios e a curva de custo × qualidade. Este limiar não é um limite técnico do modelo. Ele é uma regra de política.
- **Arquivos:** `ml/limiar.py`, `cli.py`, `tests/ml/test_limiar.py`, `tests/test_cli.py`.
- **Fazer:** seguir o passo a passo abaixo.

#### Passo a passo

1. **Usar dados rotulados próprios.**
   - Use o mesmo tokenizador que será usado em produção.


2. **Separar os dados por documento.**
   - Divida em treino, validação e teste com `dividir_por_documento`.

3. **Contar os tokens de cada pergunta.**
   - Lembre que a primeira regra do v1 usa pergunta + contexto. A segunda regra usa só a pergunta.

4. **Montar a lista de candidatos.**
   - Use os percentis da distribuição de tokens das perguntas: 50, 60, 70, 80, 90 e 95.
   - Inclua também valores fixos úteis para comparação, por exemplo 100, 200, 300 e 500.

5. **Simular o v1 para cada candidato.**
   - Para cada `T` candidato, crie uma cópia da `Config` com `limiar_tokens_pergunta=T` com as mesmas regras da camada v1.

6. **Calcular as métricas de cada candidato.**
   - Qualidade média: soma das notas dividida por `2 × número de linhas`.
   - Custo estimado: tokens de entrada e de saída multiplicados por `cfg.preco_ref`.
   - Compare com a política `sempre_forte`.
   - Calcule `qualidade_relativa_pct`, `custo_relativo_pct` e `economia_pct`.
   - Grave uma linha por candidato em `resultados/calibracao_limiar_tokens.csv`.

7. **Escolher o limiar com um critério explícito.**
   - Defina a meta em configuração, por exemplo `qualidade_minima_pct = 0.95`.
   - Escolha o maior `T` que mantém a qualidade relativa maior ou igual à meta. Um `T` maior envia menos perguntas ao forte e aumenta a economia.

8. **Salvar o resultado.**
   - Grave `resultados/config_limiar_tokens.json` com um objeto que o `carregar_config` aceite:

9. **Validar nos dados de teste.**
   - Rode `roteador avaliar --config resultados/config_limiar_tokens.json`.
   - Compare a linha `v1` com `sempre_forte` e com o valor anterior.
   - Se a qualidade no teste ficar abaixo da meta, não adote o limiar.
   

- **Pronto quando:** o comando gera um `config_limiar_tokens.json` reproduzível, com o limiar escolhido e as métricas. 
- **Aviso:** este limiar vale para o v1. O v2 usa `limiar_probabilidade`. O valor depende do tokenizador. Se o tokenizador mudar, calibre o limiar de novo.

### Camada 15: treinar `limiar_probabilidade` com RouterBench

- **Objetivo:** achar o valor de `limiar_probabilidade` que dá o melhor equilíbrio custo × qualidade, usando o [RouterBench](https://arxiv.org/html/2403.12031v1) como dados externos de calibração. A avaliação final continua nos dados próprios.
- **Arquivos:** `dados/routerbench.py`, `ml/limiar.py`, `cli.py`, `tests/dados/test_routerbench.py`, `tests/ml/test_limiar.py`, `tests/test_cli.py`.
- **Fazer:** seguir o passo a passo abaixo.

#### Passo a passo

1. **Baixar e versionar um recorte do RouterBench.**

2. **Escolher um par de modelos.**
   - Escolher um modelo barato e um modelo forte do RouterBench.

3. **Conferir as colunas do arquivo real.**

4. **Converter cada par de resultados para `LinhaDataset`.**
   - A regra de “o barato basta” continua sendo `nota_barato >= nota_forte`.

5. **Validar o dataset convertido.**
   - Conferir alguns pares à mão. Compare o prompt, os dois scores e as duas notas.

6. **Separar treino e validação por prompt.**.
   - Nunca coloque o mesmo prompt no treino e na validação.
   - Separe também um conjunto de teste. A escolha do limiar usa a validação. O teste só confirma o resultado no fim.

7. **Treinar os classificadores A e B.**
   - Use `treinar_modelo_v2` no conjunto de treino do RouterBench.
   - Use o mesmo embedder e o mesmo tokenizador que serão usados em produção.
   - Guarde o bundle com `salvar_modelo`, porque o limiar pertence ao modelo A treinado.

8. **Escolher o limiar com um critério explícito.**
    - Defina a meta em configuração, por exemplo `qualidade_minima_pct = 0.95`.
    - Escolha o menor limiar que mantém a qualidade relativa maior ou igual à meta. Isso maximiza a economia.
    - Se nenhum limiar atingir a meta, escolha o candidato de melhor qualidade e registre um aviso.
    - Em empate, prefira o limiar mais alto. Isso deixa a decisão mais conservadora.

9. **Validar o limiar nos dados próprios.**
    - Rode `roteador avaliar --config resultados/config_limiar_probabilidade.json`.
    - Compare a linha `v2` com `sempre_forte` e com o limiar anterior.
    - Se a qualidade cair abaixo da meta nos dados próprios, não adote o limiar.
    - Nesse caso, use o RouterBench apenas como pré-calibração.

10. **Adicionar o subcomando `treinar-limiar-probabilidade`.**
    - Interface sugerida:

      ```bash
      roteador treinar-limiar-probabilidade \
        --routerbench dados/routerbench/recorte.csv \
        --barato modelo-barato \
        --forte modelo-forte \
        --meta 0.95 \
        --saida resultados/config_limiar_probabilidade.json
      ```


- **Pronto quando:** o comando gera um `config_limiar_probabilidade.json` reproduzível, com o limiar escolhido e as métricas. O comando `roteador avaliar --config` aceita o arquivo. A curva mostra a política `v2` com o novo limiar.

### Camada 16: servidor HTTP local compatível com OpenAI

- **Objetivo:** expor o roteador como uma API local para clientes compatíveis com OpenAI.
- **Arquivos:** `servidor.py`, `tests/test_servidor.py`.
- **Fazer:**
  1. Criar `POST /v1/chat/completions`.
  2. Converter `messages` para `pergunta` e `contexto`.
  3. Chamar `Orquestrador.responder(...)`.
  4. Devolver a resposta no formato OpenAI.
  5. Criar `GET /health`.
  6. Ler a configuração de variáveis de ambiente ou JSON.
- **Testes:** endpoints com adapters falsos e `tmp_path`, sem rede.
- **Pronto quando:** um cliente compatível com OpenAI aponta para o servidor local e recebe a resposta roteada.
- **Limite inicial:** sem streaming e sem function calling.

## 8. Estado atual

- [x] Camada 0: esqueleto, tipos e configuração
- [x] Camada 1: tokenizador
- [x] Camada 2: contrato de adapter e adapter falso
- [x] Camada 3: adapters Ollama e Gemini
- [x] Camada 4: detector
- [x] Camada 5: registro e arquivo de ML
- [x] Camada 6: limite de tokens
- [x] Camada 7: roteador v1
- [x] Camada 8: orquestrador
- [x] Camada 9: dados e rótulos
- [x] Camada 10: avaliação e curva
- [x] Camada 11: CLI e README
- [x] Camada 12: embedder e classificadores
- [x] Camada 13: roteador v2
- [ ] Camada 14: treinar `limiar_tokens_pergunta` v1 na curva
- [ ] Camada 15: treinar `limiar_probabilidade` v2 com RouterBench
- [ ] Camada 16: servidor HTTP local compatível com OpenAI


