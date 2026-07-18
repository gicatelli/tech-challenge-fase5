# System Card — Datathon Fase 05

## Visão Geral

| Campo | Valor |
|-------|-------|
| **Nome** | Agente Inteligente para Análise de Ações (PETR4.SA) |
| **Versão** | 1.0.0 |
| **Tipo** | Sistema RAG + Agente ReAct + LSTM |
| **Domínio** | Mercado financeiro (previsão de preços, análise de risco) |
| **Proprietário** | Giovanna Catelli |
| **Fase** | Datathon Fase 05 — Pós-Tech MLET FIAP |

## Arquitetura

```
Dados yfinance → Feature Engineering → LSTM (Optuna) → MLflow Registry
                                                            ↓
Usuário → InputGuardrail → Agente ReAct (4 tools) → OutputGuardrail → Resposta
                                   ↓
                         RAG (ChromaDB + OpenAI embeddings)
                                   ↓
                     Prometheus + Grafana (observabilidade)
```

## Dados

| Fonte | Período | Registros | Tipo |
|-------|---------|:---------:|------|
| PETR4.SA (yfinance) | 2018-01 a 2026-05 | 2081 | Real |
| VALE3.SA | 2018-01 a 2026-05 | 2081 | Correlacionado |
| ITUB4.SA | 2018-01 a 2026-05 | 2081 | Correlacionado |
| Knowledge Base | - | 12 documentos | Criado manualmente |

## Modelo LSTM

| Parâmetro | Valor |
|-----------|-------|
| Arquitetura | LSTM multicamada + Dense + Dropout |
| Otimização | Optuna (30 trials, busca bayesiana) |
| Sequence length | 60 dias |
| Features | 25+ indicadores técnicos |
| Métricas (esperadas) | MAE < R$2.00, MAPE < 5% |

## Avaliação RAG (RAGAS)

| Métrica | Score (proxy) | Esperado (com OpenAI) |
|---------|:---:|:---:|
| Faithfulness | 0.98 | 0.85+ |
| Answer Relevancy | 0.63 | 0.80+ |
| Context Precision | 0.73 | 0.85+ |
| Context Recall | 0.28 | 0.75+ |

## LLM-as-Judge

| Critério | Score (proxy) | Esperado |
|----------|:---:|:---:|
| Correção factual | 2.95/5 | 4.0+/5 |
| Completude | 4.65/5 | 4.5+/5 |
| Relevância de negócio | 2.25/5 | 4.0+/5 |
| Clareza | 5.00/5 | 4.5+/5 |

## Segurança

| Camada | Implementação | Eficácia |
|--------|---------------|:--------:|
| Input Guardrail | 9 regex injection + 3 exfiltration + max_length | 100% (7/7 Red Team) |
| Output Guardrail | Presidio PII removal (CPF, email, telefone, nomes) | 100% (3/3 PII removidos) |
| Tools | Read-only, sem acesso a sistema | Verificado |
| OWASP | 7/10 ameaças mapeadas, 6 mitigadas | Documentado |

## Explicabilidade

### Feature Importance (top 5)
1. log_return_1d (24.07%)
2. log_return_5d (21.75%)
3. sma_7_ratio (11.11%)
4. volume_norm (10.22%)
5. rsi_14 (9.54%)

### Transparência do Agente
Cada interação expõe os steps intermediários:
- Thought (raciocínio)
- Action (tool escolhida)
- Observation (resultado)
- Final Answer (resposta ao usuário)

## Fairness

### Regimes de Mercado
| Regime | Performance | Observação |
|--------|:-----------:|------------|
| Bull | Sharpe 0.11 | Captura tendência de alta |
| Bear | Sharpe 1.54 | Melhor performance (contra-intuitivo — reflexo do período) |
| Neutral | Sharpe -0.34 | Dificuldade em mercados laterais |

### Viés Direcional
- Previsões: 51% up / 49% down
- **Viés**: NEUTRO — modelo não é sistematicamente otimista nem pessimista

## Monitoramento

| Componente | Ferramenta | Métricas |
|------------|-----------|----------|
| API | Prometheus | Latência P50/P95, requests, erros |
| Drift | PSI manual | Por feature, threshold 0.1/0.2 |
| Dashboard | Grafana | 7 panels pré-configurados |
| Telemetria | MLflow + arquivo local | Input, output, latência, tools |
| Alertas | 5 regras | Drift, latência, erro, degradação |

## Limitações

1. **Não prevê black swans**: eventos imprevisíveis (crises, guerras) não são capturados
2. **Mudanças de regime**: performance degrada em transições bull→bear
3. **Dados históricos**: modelo é treinado com dados passados que podem não refletir o futuro
4. **Latência**: agente com LLM tem latência de 2-5s por query
5. **Dependência de API**: requer OpenAI API key para funcionamento completo
6. **Sem dados fundamentalistas em tempo real**: usa knowledge base estática

## Riscos

| Risco | Probabilidade | Impacto | Mitigação |
|-------|:---:|:---:|-----------|
| Previsão incorreta causa perda financeira | Alta | Alto | Disclaimer obrigatório |
| Drift não detectado | Média | Alto | Monitoramento PSI a cada 6h |
| Prompt injection bypass | Baixa | Médio | 9 patterns + atualização contínua |
| Indisponibilidade da API OpenAI | Média | Médio | Fallback para tools diretas |
