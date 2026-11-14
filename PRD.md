# PRD — Roteador de LLMs (custo × qualidade)

| Campo | Valor |
| --- | --- |
| **Produto** | Roteador de LLMs (custo × qualidade) |
| **Versão do documento** | 1.0 |
| **Versão do produto** | 0.1.0 (MVP) |
| **Status** | Em desenvolvimento. Camadas 0–13 implementadas. Camadas 14–16 planejadas. |
| **Última atualização** | 2026-11-13 |
| **Público** | Engenharia, produto, pesquisa e avaliação técnica |
| **Documentos relacionados** | [README](README.md) e [SPEC](SPEC.md) |

## 1. Resumo executivo

O Roteador de LLMs decide, para cada pergunta, se usa um modelo **barato** ou um modelo **forte**. O objetivo é reduzir custo sem perder a maior parte da qualidade do modelo forte.

O sistema mede a decisão. Cada requisição gera um registro com modelo, tokens, tempo, custo estimado e motivo da decisão. O dataset rotulado permite comparar políticas em uma curva de custo × qualidade.

O resultado esperado é uma frase objetiva:

> “Mantive **X%** da qualidade do modelo forte gastando **Y%** menos.”

## 2. Contexto e problema

### 2.1 Problema

Produtos com LLM pagam por token. Modelos fortes custam muito mais que modelos pequenos. A maior parte das perguntas é simples. Um modelo pequeno responde bem a essas perguntas.

Enviar tudo ao modelo forte desperdiça dinheiro e aumenta a latência. Enviar tudo ao modelo pequeno reduz a qualidade nas perguntas difíceis.

### 2.2 Oportunidade

A decisão pode ser feita por requisição. Assim, o sistema usa o modelo barato quando ele basta e reserva o modelo forte para os casos difíceis.

### 2.3 Alternativas atuais

| Estratégia | Custo | Qualidade | Problema |
| --- | --- | --- | --- |
| Sempre barato | Mínimo | Baixa nas perguntas difíceis | Não atende perguntas complexas |
| Sempre forte | Máximo | Alta | Custo desnecessário na maioria das perguntas |
| Regra fixa sem dados | Baixo | Imprevisível | Não há evidência de custo × qualidade |
| **Roteador com avaliação** | **Variável** | **Controlada por meta** | **Escolha por requisição com medição** |

## 3. Objetivos

### 3.1 Objetivo principal

Reduzir o custo estimado das respostas mantendo uma meta de qualidade em relação ao modelo forte.

### 3.2 Objetivos específicos

- Rotear cada requisição entre um modelo barato e um modelo forte.
- Registrar modelo, tokens, tempo, custo estimado e motivo da decisão.
- Gerar uma curva de custo × qualidade sobre um dataset rotulado.
- Comparar políticas: sempre barato, sempre forte, aleatório, perfeito, v1, forte limitado e v2.
- Detectar respostas vazias ou truncadas e tratar cada caso com no máximo uma nova tentativa.
- Calibrar limiares com dados, não com valores fixos sem evidência.

### 3.3 Não objetivos

- Treinar ou ajustar um LLM.
- Criar uma interface web.
- Implementar autenticação e cobrança.
- Rotear entre mais de dois modelos nesta versão.
- Garantir custo real de provedor. O custo é uma estimativa por preço de referência.

## 4. Usuários e cenários

### 4.1 Personas

| Persona | Necessidade |
| --- | --- |
| Engenheiro de aplicação | Reduzir custo de LLM sem reescrever o produto |
| Pesquisador de ML | Comparar políticas e calibrar limiares com dados |
| Produto | Entender o equilíbrio entre custo e qualidade |
| Operação | Monitorar respostas vazias, truncadas e falhas de comunicação |

### 4.2 Histórias de usuário

- **US-01:** Como engenheiro, quero enviar uma pergunta pelo roteador para receber a resposta do modelo escolhido.
- **US-02:** Como engenheiro, quero aplicar um limite de tokens ao modelo forte para reduzir o custo de respostas médias.
- **US-03:** Como pesquisador, quero treinar o roteador v2 com dados rotulados para melhorar a decisão.
- **US-04:** Como pesquisador, quero avaliar políticas offline para escolher a melhor relação custo × qualidade.
- **US-05:** Como operação, quero monitorar respostas vazias e truncadas para detectar falhas.
- **US-06:** Como usuário de ferramentas compatíveis com OpenAI, quero apontar um cliente para um servidor local e obter respostas roteadas. *(Planejado.)*

## 5. Proposta de solução

### 5.1 Fluxo principal

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
        └──────────────► modelo forte (Gemini)
        │
        ▼
  orquestrador + adapter
        │
        ▼
  detector de respostas vazias ou truncadas
        │
        ▼
  resposta + registro + arquivo de ML
```

### 5.2 Roteadores

| Versão | Mecanismo | Entrada | Saída |
| --- | --- | --- | --- |
| **v1** | Regra determinística | Tokens da pergunta e do contexto | Barato ou forte |
| **v2** | Dois classificadores de ML | Embedding da pergunta e `log1p` dos tokens | Barato ou forte + faixa prevista |

O v2 tem dois classificadores:

- **A (qualidade):** “o modelo barato basta?”. Decide o roteamento.
- **B (tamanho):** prevê `curta`, `media` ou `longa`. Define o limite no modelo forte.

### 5.3 Limite de tokens

Respostas de faixa média vão ao modelo forte com um limite `X`. O prompt recebe o aviso “Limite seus gastos a cerca de X tokens”. Um teto rígido maior que `X` evita cortar a resposta no meio.

### 5.4 Respostas vazias ou truncadas

O sistema classifica a resposta. Se ela for vazia ou truncada, há no máximo uma nova tentativa:

- o forte repete com o teto dobrado;
- o barato escala para o forte, sem limite.

Cada ocorrência também vai para o arquivo de ML.

### 5.5 Avaliação

A avaliação roda sobre o dataset já gravado. Ela compara políticas e gera a curva de custo × qualidade. O resultado principal mostra a qualidade relativa e a economia em relação ao modelo forte.

## 6. Escopo

### 6.1 Escopo entregue no MVP

- Camadas 0–13 da [SPEC](SPEC.md).
- Roteador v1 determinístico.
- Roteador v2 com classificadores A e B.
- Adapters Ollama e Gemini.
- Detector de respostas vazias ou truncadas.
- Retry e fallback de comunicação.
- Registro em CSV.
- Dataset em CSV e SQLite.
- Avaliação com curva e relatório de alertas.
- CLI com `rotear`, `coletar`, `avaliar` e `treinar`.
- Suíte de testes sem rede.

### 6.2 Incrementos planejados

| Camada | Incremento | Resultado esperado |
| --- | --- | --- |
| 14 | Calibrar `limiar_tokens_pergunta` na curva | Limiar do v1 com evidência |
| 15 | Calibrar `limiar_probabilidade` com RouterBench | Limiar do v2 com dados externos |
| 16 | Servidor HTTP local compatível com OpenAI | Uso por clientes externos |

### 6.3 Fora de escopo

- Ajuste fino de modelos.
- Interface web e autenticação.
- Mais de dois modelos no roteamento.
- Streaming e function calling no servidor HTTP inicial.
- Cobrança e controle financeiro real.
- Interceptação automática de prompts de outros programas.

## 7. Requisitos

### 7.1 Requisitos funcionais

| ID | Requisito | Critério de aceite |
| --- | --- | --- |
| **RF-01** | Rotear cada pergunta entre barato e forte | O resultado contém papel, motivo, faixa e limite |
| **RF-02** | Levar ao forte quando o contexto não cabe no barato | A regra compara pergunta + contexto com a janela |
| **RF-03** | Aplicar limite de tokens no forte | O prompt e o teto usam o mesmo `X` |
| **RF-04** | Repetir falhas transitórias | Rede, HTTP 408/425/429/5xx e JSON inválido são repetidos |
| **RF-05** | Fazer fallback entre papéis | O outro papel é tentado quando o principal falha |
| **RF-06** | Detectar resposta vazia ou truncada | A resposta recebe `vazio`, `truncado` ou `possivelmente_truncado` |
| **RF-07** | Fazer no máximo uma nova tentativa | Não existe laço de repetição |
| **RF-08** | Registrar cada requisição | O CSV grava decisão, tokens, tempo e custo |
| **RF-09** | Gravar ocorrências de ML | O CSV de ML grava tipo, ação e resultado final |
| **RF-10** | Ler dataset em CSV ou SQLite | O carregador escolhe pela extensão |
| **RF-11** | Separar treino e teste por documento | O mesmo documento não aparece nos dois lados |
| **RF-12** | Avaliar políticas offline | A curva gera qualidade, custo e percentuais |
| **RF-13** | Treinar o v2 | O modelo A e o modelo B são salvos com `joblib` |
| **RF-14** | Expor uma CLI | Os subcomandos funcionam com `--config` |
| **RF-15** | Expor um servidor HTTP compatível com OpenAI | *Planejado na camada 16* |

### 7.2 Requisitos não funcionais

| ID | Requisito | Critério de aceite |
| --- | --- | --- |
| **RNF-01** | Python 3.11 ou superior | A instalação funciona no ambiente declarado |
| **RNF-02** | Testes sem rede | `pytest` passa sem Ollama e sem Gemini |
| **RNF-03** | Baixo acoplamento | O orquestrador depende de interfaces, não de provedores |
| **RNF-04** | Reprodutibilidade | Sementes e caminhos ficam na configuração |
| **RNF-05** | Segurança | Nenhuma chave é gravada no repositório |
| **RNF-06** | Observabilidade | Cada decisão pode ser auditada no registro |
| **RNF-07** | Armazenamento substituível | Um novo `Registro` pode substituir o CSV |
| **RNF-08** | Documentação | README explica instalação e uso; SPEC explica as camadas |

## 8. Métricas de sucesso

As metas abaixo são alvos do projeto. Elas devem ser medidas no conjunto de teste, nunca nos dados de calibração.

| Métrica | Meta inicial | Como medir |
| --- | --- | --- |
| Retenção de qualidade | ≥ 95% do `sempre_forte` | Curva de custo × qualidade |
| Redução de custo | ≥ 50% em relação ao `sempre_forte` | Custo estimado por `preco_ref` |
| Taxa de respostas vazias | < 5% | Relatório de alertas |
| Taxa de respostas truncadas | < 5% | Relatório de alertas |
| Cobertura do fluxo | 100% dos caminhos principais | Testes com adapters falsos |
| Reprodutibilidade | Mesmo resultado com a mesma semente | Repetição da avaliação |

## 9. Arquitetura e decisões

### 9.1 Componentes

| Componente | Responsabilidade |
| --- | --- |
| `Tokenizador` | Contar tokens |
| `Adapter` | Interface comum para Ollama e Gemini |
| `RoteadorV1` | Decidir por tokens |
| `RoteadorV2` | Decidir por classificadores |
| `Orquestrador` | Executar o fluxo completo |
| `Detector` | Classificar respostas vazias ou truncadas |
| `Registro` | Gravar requisições e ocorrências |
| `Avaliação` | Comparar políticas e gerar a curva |
| `CLI` | Ponto de entrada por terminal |

### 9.2 Decisões principais

| Decisão | Motivo | Consequência |
| --- | --- | --- |
| Dois modelos | Reduzir custo com controle de qualidade | Menos flexibilidade que um roteador multi-modelo |
| v1 antes do v2 | Entregar um plano mínimo funcional | v1 não entende o conteúdo da pergunta |
| Adapters com interface única | Trocar provedor sem mudar a lógica | Cada adapter normaliza tokens e parada |
| CSV no MVP | Simples de auditar | Migração para banco exige outro adapter |
| Preço de referência | Modelos usados são gratuitos | O custo é estimado, não cobrado |
| Separação por documento | Evitar vazamento de dados | Exige pelo menos dois documentos |

## 10. Dados e avaliação

### 10.1 Dataset

```text
id,documento,tipo,pergunta,contexto_tokens,nota_barato,nota_forte,tokens_resposta_barato,tokens_resposta_forte,truncada_forte,nota_forte_limitado,tokens_resposta_forte_limitado
```

Regras:

- Notas: `correta` (2), `parcial` (1) ou `errada` (0).
- “O barato basta” quando `nota_barato >= nota_forte`.
- Resposta truncada sobe uma faixa de tamanho.
- A divisão treino/teste é feita por documento.

### 10.2 Qualidade dos dados

- Incluir perguntas diretas, perguntas que exigem juntar partes, perguntas sem resposta no contexto e perguntas ambíguas.
- Preencher as notas de forma consistente.
- Separar dados de calibração e dados de teste.
- Documentar a origem e a versão do dataset.

### 10.3 RouterBench

O RouterBench pode apoiar a calibração do `limiar_probabilidade` na camada 15. Ele não substitui a avaliação final nos dados próprios.

## 11. Riscos e mitigação

| Risco | Impacto | Mitigação |
| --- | --- | --- |
| Dataset pequeno | Métrica instável | Ampliar o dataset e separar por documento |
| Rótulos inconsistentes | Decisão ruim | Definir critérios e revisar amostras |
| Vazamento entre treino e teste | Resultado inflado | Dividir por documento |
| Limiar calibrado no teste | Resultado otimista | Usar validação separada |
| Diferença de domínio no RouterBench | Limiar não transfere | Validar nos dados próprios |
| Cota do Gemini | Falha de comunicação | Retry, fallback e aviso claro |
| Ollama indisponível | Barato e embeddings falham | Mensagem clara e fallback |
| Custo estimado diferente do real | Métrica enganosa | Atualizar `preco_ref` e documentar |
| Pergunta curta e difícil | v1 escolhe o barato | Usar v2 com dados rotulados |
| Resposta truncada | Qualidade e custo ruins | Detector e nova tentativa |
