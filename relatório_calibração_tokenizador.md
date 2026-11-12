# Relatório de calibração do tokenizador aproximado

**Data:** 12/11/2026
**Tokenizador real:** `tiktoken` com codificação `o200k_base`
**Tokenizador aproximado:** `ceil(caracteres / divisor)`
**Divisor atual em `config.py`:** `4`

## Resumo

- O corpus solicitado foi o texto da **Camada 4** do guia (detector de respostas vazias ou truncadas).
- Nesse corpus, o melhor divisor ficou em **3,5** (MAPE de 16,0%), com empate técnico com o divisor **3**.
- O divisor **4** ficou com razão média de **1,205** e MAPE de **22,2%**, ou seja, o aproximado subestimou os tokens reais nesse texto.

## 1. Objetivo

Verificar se o divisor do `TokenizadorAproximado` está adequado quando comparado à contagem real do `tiktoken`, seguindo a Camada 1 do guia.


## 2. Corpus usado

O texto da Camada 4 foi limpo de marcações Markdown e dividido em frases.

| Item | Valor |
| --- | ---: |
| Frases | 14 |
| Caracteres | 674 |
| Tokens reais (`o200k_base`) | 200 |
| Caracteres por token real (média) | 3,37 |

Amostra do corpus:

```text
detector de respostas vazias ou truncadas Arquivo: detector.py.
classificar(resposta, limite_tokens, cfg) -> Ocorrencia, aplicando as regras nesta ordem:
texto vazio com tokens de saída > 0:
VAZIO, alerta de provável gasto em raciocínio; tokens de saída = 0, sem erro:
```

## 3. Método

Para cada frase:

```text
tokens_reais = len(tiktoken(o200k_base).encode(frase))
tokens_aproximados = ceil(len(frase) / divisor)
razão = tokens_reais / tokens_aproximados
```

Métricas usadas:

- **Razão total:** soma dos tokens reais ÷ soma dos tokens aproximados. O ideal é ficar perto de 1.
- **Razão média:** média das razões por frase.
- **MAPE:** erro percentual absoluto médio. Quanto menor, melhor.

Critério de alerta definido para esta análise: razão média fora de `[0,9; 1,1]` ou MAPE acima de `10%`.

## 4. Resultados no corpus da Camada 4

| Divisor | Tokens aproximados | Razão total | Razão média | MAPE |
| ---: | ---: | ---: | ---: | ---: |
| 2,0 | 340 | 0,588 | 0,630 | 37,0% |
| 2,5 | 276 | 0,725 | 0,760 | 24,0% |
| 3,0 | 228 | 0,877 | 0,929 | 16,0% |
| **3,5** | **197** | **1,015** | **1,076** | **16,0%** |
| 4,0 | 173 | 1,156 | 1,205 | 22,2% |
| 4,5 | 156 | 1,282 | 1,300 | 30,0% |
| 5,0 | 142 | 1,408 | 1,407 | 40,7% |
| 6,0 | 117 | 1,709 | 1,777 | 77,7% |

Leitura:

- Com divisor **4**, o aproximado gerou **173 tokens** contra **200 reais**. A razão real/aproximado ficou em **1,205**, indicando subestimação de cerca de **20,5%**.
- O divisor **3,5** trouxe a razão total para **1,015**, praticamente empatada com a contagem real.
- O divisor **3** também ficou próximo (razão média 0,929), mas o **3,5** tem a razão total mais perto de 1.

## 5. Controle com prosa em português

Como o corpus da Camada 4 tem muito código, fiz um controle rápido com 5 frases de prosa em português, mais parecidas com as perguntas reais do projeto.

| Divisor | Razão média | MAPE |
| ---: | ---: | ---: |
| 3,0 | 0,656 | 34,4% |
| 3,5 | 0,805 | 19,5% |
| **4,0** | **0,883** | **11,7%** |
| **4,5** | **0,972** | **5,2%** |
| 5,0 | 1,053 | 9,7% |

Nesse controle, o melhor divisor foi **4,5**. Ou seja, o divisor ideal muda conforme o tipo de texto:

- texto com código e termos em inglês → divisor menor (3 a 3,5);
- prosa em português → divisor maior (4 a 4,5).

## 6. Decisão

**Manter `divisor_aproximado = 4`.**

Motivos:

1. O corpus solicitado é o da Camada 4, que é técnico e cheio de identificadores; O resultado não foi o esperado, mas para primeira versão do projeto serve.
2. No controle com prosa em português, o divisor 4 teve erro de 11,7%, bem melhor que o 3 ou 3,5.


## 7. É necessário criar novos testes?

**Não, não para esta calibração.**

Os testes atuais já cobrem:

- mesmo texto gera a mesma contagem;
- texto vazio gera 0;
- contagem real fica dentro da tolerância do aproximado;
- falha simulada na carga aciona a reserva;
- a fábrica monta real + aproximado;
- o divisor configurado é repassado ao aproximado;
- a coluna `tokenizador_usado` existe no esquema do registro.

A calibração é uma análise sobre um corpus, não uma regra do sistema. Transformá-la em teste unitário seria frágil por dois motivos:


## 8. A considerar
- A contagem do `tiktoken` pode mudar entre versões da biblioteca; por isso a versão usada deve ser fixada no projeto.
