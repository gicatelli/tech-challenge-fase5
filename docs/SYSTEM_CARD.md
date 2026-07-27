# System Card — Datathon Fase 05

## Visão Geral

| Campo | Valor |
|-------|-------|
| **Nome** | Agente Inteligente para Análise de Ações (PETR4.SA) |
| **Versão** | 1.1.0 |
| **Tipo** | Sistema RAG + Agente ReAct + Ensemble (LSTM + RF) |
| **Domínio** | Mercado financeiro (previsão de preços, análise de risco) |
| **Proprietário** | Giovanna Catelli |
| **Fase** | Datathon Fase 05 — Pós-Tech MLET FIAP |
| **Data de execução** | 27 de julho de 2026 |

## Arquitetura

```
Dados yfinance (PETR4 + VALE3 + ITUB4 + Brent + USD/BRL + Ibovespa)
    ↓
Feature Engineering (37 features: técnicas + exógenas + correlação)
    ↓
Ensemble (LSTM Optuna + Random Forest) → MLflow Registry
    ↓
Usuário → InputGuardrail → Agente ReAct (4 tools) → OutputGuardrail → Resposta
                                   ↓
                    RAG (ChromaDB + embeddings locais + re-ranking)
                                   ↓
                  Prometheus + Grafana (observabilidade + drift)
```

## Dados

| Fonte | Período | Registros | Tipo |
|-------|---------|:---------:|------|
| PETR4.SA (yfinance) | 2020-01-02 a 2026-07-24 | 1634 | Real (target) |
| VALE3.SA (yfinance) | 2020-01-02 a 2026-07-24 | 1634 | Feature correlação |
| ITUB4.SA (yfinance) | 2020-01-02 a 2026-07-24 | 1634 | Feature correlação |
| Brent, USD/BRL, Ibovespa | 2020-01-02 a 2026-07-24 | ~1634 | Features exógenas |
| Knowledge Base | — | 17 documentos | Criado manualmente |
| Golden Set (RAGAS) | — | 32 pares | Avaliação RAG |

## Modelo Champion: LSTM

| Parâmetro | Valor |
|-----------|-------|
| Arquitetura | LSTM multicamada + Dense + Dropout |
| Otimização | Optuna (30 trials, busca bayesiana) |
| Sequence length | 60 dias |
| Features | 27 indicadores técnicos (+ exógenas + correlação) |
| Épocas (max) | 50 (com early stopping, patience=10) |
| Batch size | 32 |

### Métricas do Modelo (conjunto de teste, valores reais em R$)

| Modelo | MAE | RMSE | MAPE | R² | Latência |
|--------|:---:|:----:|:----:|:--:|:--------:|
| **LSTM** | R$ 4.07 | R$ 5.82 | 10.8% | 0.2510 | 0.02ms |
| Random Forest | R$ 3.87 | R$ 6.21 | 9.1% | 0.1480 | 0.11ms |
| **Ensemble (LSTM+RF)** | — | < 5.82 (esperado) | — | > 0.25 (esperado) | — |

**Champion selecionado: LSTM** (RMSE 6.3% menor que RF, R² 70% superior)

## Avaliação RAG (RAGAS) — Valores Reais

Avaliado com golden set de 32 pares, LLM local Qwen2.5:3b, embeddings all-MiniLM-L6-v2.

| Métrica | Score | Classificação |
|---------|:-----:|:---:|
| **Faithfulness** | 0.7035 | Moderado |
| **Answer Relevancy** | 0.7226 | Moderado |
| **Context Precision** | 0.5553 | Precisa melhorar |
| **Context Recall** | 0.6209 | Aceitável |

**Nota:** Scores obtidos com modelo local de 3B params em CPU. Com modelo maior (GPT-4o ou Qwen2.5:14b), scores esperados > 0.80. Melhorias implementadas (chunk 500 + re-ranking) devem elevar context_precision em próxima execução.

## LLM-as-Judge — Valores Reais

Avaliado com 4 critérios + overall, usando Qwen2.5:3b como judge.

| Critério | Score |
|----------|:-----:|
| Correção factual | 3.95/5.0 |
| Completude | 3.10/5.0 |
| Relevância de negócio | 3.30/5.0 |
| Clareza | 4.40/5.0 |
| **Overall** | **3.69/5.0** |

## Segurança

| Camada | Implementação | Eficácia |
|--------|---------------|:--------:|
| Input Guardrail | 9 regex injection + 3 exfiltration + max_length | 100% (7/7 Red Team) |
| Output Guardrail | Presidio PII + regex (CPF, email, telefone, cartão) | 100% (PII sanitizado) |
| Tools | Read-only, sem acesso a sistema | Verificado |
| OWASP | 7/10 ameaças mapeadas, 6 com mitigação implementada | Documentado |

### Red Team Results

| # | Cenário | Status |
|---|---------|:------:|
| RT-01 | Prompt Injection | BLOQUEADO |
| RT-02 | Data Exfiltration | BLOQUEADO |
| RT-03 | Jailbreak (DAN) | BLOQUEADO |
| RT-04 | PII Leakage | PERMITIDO (legítimo) |
| RT-05 | Tool Manipulation | BLOQUEADO |
| RT-06 | Context Overflow (DoS) | BLOQUEADO |
| RT-07 | Indirect Injection (forget) | BLOQUEADO |

## Drift Detection — Status Atual

| Métrica | Valor |
|---------|:-----:|
| Status | CRITICAL |
| Max PSI | 1.5667 (macd) |
| Ação automática | Retrain disparado |
| Resultado retrain | Champion MANTIDO (challenger 6.79% pior) |

### PSI por Feature

| Feature | PSI | Status |
|---------|:---:|:------:|
| macd | 1.5667 | CRITICAL |
| volatility_30 | 1.1183 | CRITICAL |
| bb_width | 0.2540 | CRITICAL |
| rsi_14 | 0.1691 | WARNING |
| price_sma30_ratio | 0.1237 | WARNING |
| log_return | 0.0560 | OK |
| volume_norm | 0.0308 | OK |

## Explicabilidade

### Feature Importance (top 5 — Permutation Importance)
1. log_return_1d (24.07%)
2. log_return_5d (21.75%)
3. sma_7_ratio (11.11%)
4. volume_norm (10.22%)
5. rsi_14 (9.54%)

### Transparência do Agente ReAct
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
| Bear | Sharpe 1.54 | Melhor performance (modelo captura reversões) |
| Neutral | Sharpe -0.34 | Dificuldade em mercados laterais |

### Viés Direcional
- Previsões: 51% up / 49% down
- **Viés**: NEUTRO — modelo não é sistematicamente otimista nem pessimista

## Monitoramento e Observabilidade

| Componente | Ferramenta | Status |
|------------|-----------|:------:|
| API | FastAPI (8 endpoints) | Healthy, < 100ms |
| MLflow | SQLite local | Online, 52 runs |
| Drift | PSI customizado | CRITICAL (retrain executado) |
| Dashboard | Gerado via matplotlib | 6 painéis |
| Testes | pytest (178 passed) | Coverage 64% |
| CI/CD | GitHub Actions | lint + mypy + bandit + pytest |
| Guardrails | Input + Output | 7/7 bloqueados |

## RAG Pipeline

| Componente | Configuração |
|------------|-------------|
| Embedding model | all-MiniLM-L6-v2 (local, gratuito) |
| Vector store | ChromaDB (persistido) |
| Chunk size | 500 chars |
| Chunk overlap | 100 chars |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Retrieval | 10 candidatos → re-rank → top 3 |
| Knowledge base | 17 documentos (glossários, análises, regulação) |
| LLM | Qwen2.5:3b (Ollama, local, gratuito) |

## Limitações

1. **Não prevê black swans**: eventos imprevisíveis (crises, guerras) não são capturados
2. **Mudanças de regime**: performance degrada em transições bull→bear (drift detectado)
3. **R² = 0.25**: modelo explica apenas 25% da variância — útil como suporte, não oráculo
4. **Latência do agente**: ~11.5s por query com LLM local (requer GPU para < 2s)
5. **Context Precision 0.55**: quase metade dos chunks recuperados são ruído (melhorias em andamento)
6. **LLM local limitado**: Qwen2.5:3b tem capacidade inferior a GPT-4o — trade-off custo zero vs qualidade

## Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação Implementada |
|-------|:---:|:---:|-----------|
| Previsão incorreta causa perda | Alta | Alto | Disclaimer, modelo como suporte à decisão |
| Drift não detectado | Baixa | Alto | PSI a cada 6h + alerta automático |
| Prompt injection bypass | Baixa | Médio | 9 patterns + red team + atualização contínua |
| LLM gera informação falsa | Média | Alto | Faithfulness 0.70 + supervisão humana |
| Vazamento de PII | Baixa | Crítico | Presidio + regex dupla camada |
| Indisponibilidade | Baixa | Médio | Health check + fallback tools diretas |

## Evolução Planejada

| Item | Impacto Esperado |
|------|:---:|
| Migrar para Qwen2.5:14b ou GPT-4o | RAGAS > 0.80, Judge > 4.0 |
| Re-ranking implementado (cross-encoder) | Context Precision +15-20% |
| Features exógenas (Brent, USD/BRL, Ibovespa) | R² +0.03-0.05 |
| Ensemble LSTM+RF | RMSE -3-5% |
| Expandir knowledge base (17 → 50 docs) | Context Recall +10% |
