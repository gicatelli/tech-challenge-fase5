# System Card — Datathon Fase 05

## Visão Geral do Sistema

| Campo | Valor |
|-------|-------|
| Nome do Sistema | Assistente Inteligente Datathon |
| Versão | 1.0.0 |
| Tipo | Sistema de IA com LLM + RAG + Agente |
| Domínio | Mercado Financeiro |
| Responsável | Grupo XX — FIAP Pós-Tech MLET |
| Data de Criação | 2025-XX-XX |

## Propósito e Escopo

### Objetivo
<!-- Descreva o objetivo do sistema no contexto da empresa convidada -->
O sistema foi desenvolvido para [DESCREVER OBJETIVO] utilizando técnicas de
Retrieval-Augmented Generation (RAG) e agentes inteligentes.

### Casos de Uso Pretendidos
1. [Caso de uso 1]
2. [Caso de uso 2]
3. [Caso de uso 3]

### Casos de Uso Não Pretendidos
- O sistema NÃO deve ser usado para [...]
- O sistema NÃO substitui decisão humana em [...]

## Arquitetura

### Componentes Principais
1. **Pipeline de Dados** — Ingestão, feature engineering, validação
2. **Modelo Baseline** — Random Forest + MLP PyTorch
3. **LLM + RAG** — Retrieval-Augmented Generation com ChromaDB
4. **Agente ReAct** — Agente com ≥ 3 tools customizadas
5. **API FastAPI** — Endpoint documentado com guardrails
6. **Monitoramento** — Prometheus + Grafana + Evidently

### Stack Tecnológica
- Python 3.11
- Scikit-Learn, PyTorch (modelos)
- LangChain, OpenAI (LLM/Agente)
- ChromaDB (vector store)
- FastAPI (serving)
- MLflow (experiment tracking)
- Evidently (drift detection)
- Prometheus + Grafana (observabilidade)
- Docker + Docker Compose (containerização)
- GitHub Actions (CI/CD)

## Dados

### Dados de Treinamento
- **Fonte**: [Empresa convidada]
- **Volume**: [X registros]
- **Período**: [Data início — Data fim]
- **Versionamento**: DVC

### Dados Sensíveis (LGPD)
- **PII identificado**: [Listar campos com dados pessoais]
- **Tratamento**: Anonimização via Presidio
- **Base legal**: [Consentimento / Legítimo interesse / ...]

## Modelo

### Baseline (Scikit-Learn)
- **Tipo**: Random Forest Classifier
- **Métricas**: AUC=X.XX, F1=X.XX, Precision=X.XX, Recall=X.XX
- **Features**: [Número] features

### Baseline (PyTorch)
- **Tipo**: MLP (Multi-Layer Perceptron)
- **Arquitetura**: [64, 32, 16] + Dropout 0.3
- **Métricas**: AUC=X.XX, F1=X.XX

### LLM
- **Modelo**: GPT-4o-mini
- **Quantização**: [Se aplicável]
- **Temperatura**: 0.0

## Avaliação

### RAGAS (4 métricas)
| Métrica | Score |
|---------|-------|
| Faithfulness | X.XX |
| Answer Relevancy | X.XX |
| Context Precision | X.XX |
| Context Recall | X.XX |

### LLM-as-Judge (≥ 3 critérios)
| Critério | Score (1-5) |
|----------|-------------|
| Correção Factual | X.X |
| Completude | X.X |
| Relevância de Negócio | X.X |
| Clareza | X.X |

## Segurança

### OWASP Top 10 para LLMs
Ver documento detalhado: [OWASP_MAPPING.md](./OWASP_MAPPING.md)

### Guardrails Implementados
- **Input**: Detecção de prompt injection, limite de tamanho, exfiltração
- **Output**: Remoção de PII, validação de padrões de sistema

### Red Teaming
Ver relatório: [RED_TEAM_REPORT.md](./RED_TEAM_REPORT.md)

## Fairness e Viés

### Análise de Viés
- [Descrever análise de viés realizada]
- [Grupos protegidos considerados]
- [Métricas de fairness calculadas]

### Mitigações
- [Ações tomadas para mitigar viés]

## Explicabilidade

### Métodos Utilizados
- Feature importance (Random Forest)
- [SHAP / LIME se aplicável]
- Transparência do agente (steps intermediários visíveis)

## Limitações Conhecidas

1. [Limitação 1]
2. [Limitação 2]
3. [Limitação 3]

## Monitoramento em Produção

### Métricas Operacionais
- Latência P50, P95, P99
- Taxa de erro
- Throughput (requests/s)

### Métricas de Qualidade
- Drift detection (PSI)
- Faithfulness score (Langfuse)
- User feedback

### Alertas
- PSI > 0.1 → Warning
- PSI > 0.2 → Retrain trigger
- Latência P99 > 5s → Alerta operacional
- Error rate > 5% → Alerta crítico

## Conformidade

### LGPD
Ver plano completo: [LGPD_PLAN.md](./LGPD_PLAN.md)

### Governança
- Model Registry com versionamento obrigatório
- Approval gates antes de produção
- Audit trail completo via MLflow
