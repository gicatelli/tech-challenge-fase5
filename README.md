# Datathon Fase 05 — FIAP Pós-Tech MLET

## Visão Geral

Sistema de IA com LLM, RAG e Agente ReAct para análise e previsão de preços de ações (PETR4.SA).
Projeto Integrador (Fases 01–05) do Datathon da Pós-Tech em Machine Learning Engineering.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│              Etapa 1: Dados + Baseline (Fases 01-02)         │
│  [yfinance] → [EDA] → [Feature Eng.] → [LSTM+RF] → [MLflow]│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│             Etapa 2: LLM + Agente + RAG (Fases 03-05)        │
│  [ChromaDB] → [Agente ReAct 4 tools] → [FastAPI] → [CI/CD]  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│          Etapa 3: Avaliação + Observabilidade (Fases 03-05)  │
│  [RAGAS 4 métricas] → [LLM Judge] → [Prometheus+Grafana]    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│          Etapa 4: Segurança + Governança (Fases 04-05)       │
│  [Guardrails] → [OWASP 7/10] → [Red Team 7/7] → [LGPD]     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Pré-requisitos
- Python 3.11+
- Docker + Docker Compose (para MLflow/Prometheus/Grafana)
- OpenAI API key (para LLM/RAG)

### Instalação

```bash
# Clonar
git clone https://github.com/gicatelli/tech-challenge-fase5.git
cd tech-challenge-fase5

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -e ".[dev]"

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com OPENAI_API_KEY=sk-...

# Instalar pre-commit hooks
pre-commit install
```

### Executar

```bash
# Subir infraestrutura (MLflow, Prometheus, Grafana)
docker-compose up -d

# Coletar dados
python src/data_collection.py

# Treinar modelos (LSTM + Random Forest)
python -m src.models.train

# Ingerir documentos no RAG
python -m src.agent.rag_pipeline ingest

# Iniciar API
python -m src.serving.app

# Rodar testes
pytest tests/

# Lint
ruff check src/ tests/ evaluation/
```

### Avaliação

```bash
# RAGAS (4 métricas)
python -m evaluation.ragas_eval

# LLM-as-Judge (4 critérios)
python -m evaluation.llm_judge

# Drift Detection
python -m src.monitoring.drift

# A/B Test (3 configs)
python -m evaluation.ab_test_prompts

# Red Team (7 cenários)
python scripts/run_red_team.py
```

## Métricas

### Modelo LSTM (Fase 4 → Fase 5)
| Métrica | Fase 4 (manual) | Fase 5 (Optuna) |
|---------|:---:|:---:|
| MAE | R$ 1.74 | < R$ 1.50 (esperado) |
| RMSE | R$ 2.27 | < R$ 2.00 (esperado) |
| MAPE | 4.87% | < 4.0% (esperado) |

### RAG (RAGAS — 4 métricas)
| Métrica | Score (proxy) | Com OpenAI (esperado) |
|---------|:---:|:---:|
| Faithfulness | 0.98 | 0.85+ |
| Answer Relevancy | 0.63 | 0.80+ |
| Context Precision | 0.73 | 0.85+ |
| Context Recall | 0.28 | 0.75+ |

### LLM-as-Judge (4 critérios, notas 1-5)
| Critério | Score |
|----------|:---:|
| Correção factual | 2.95 |
| Completude | 4.65 |
| Relevância de negócio | 2.25 |
| Clareza | 5.00 |

### Drift Detection
| Feature | PSI | Status |
|---------|:---:|--------|
| volatility_30 | 1.14 | CRITICAL |
| rsi_14 | 0.11 | WARNING |
| log_return | 0.07 | OK |

### Segurança
- **Red Team**: 7/7 cenários bloqueados
- **Guardrails**: 25/25 testes passando (input + output)
- **OWASP**: 7/10 ameaças mapeadas, 6 mitigadas

## Estrutura do Projeto

```
tech-challenge-fase5/
├── .github/workflows/ci.yml    # CI/CD (lint + mypy + bandit + pytest)
├── data/
│   ├── raw/                    # Dados brutos (DVC)
│   ├── processed/              # Features processadas
│   ├── golden_set/             # 20 pares para avaliação
│   └── knowledge_base/        # 12 documentos para RAG
├── src/
│   ├── features/               # Feature engineering (25 indicadores)
│   ├── models/                 # LSTM + RF + Optuna + Registry
│   ├── agent/                  # Agente ReAct + RAG + 4 tools
│   ├── serving/                # FastAPI (8 endpoints)
│   ├── monitoring/             # Drift + Métricas + Telemetria
│   └── security/               # Guardrails + PII detection
├── tests/                      # 67 testes (pytest)
├── evaluation/                 # RAGAS + LLM Judge + A/B Test
├── docs/                       # System Card, Model Card, LGPD, OWASP, Red Team
├── configs/                    # YAML configs + Grafana dashboard
├── scripts/                    # Scripts de execução e testes locais
├── docker-compose.yml          # MLflow + Prometheus + Grafana
├── pyproject.toml              # Dependências + configs
└── Makefile                    # Atalhos
```

## Documentação

| Documento | Conteúdo |
|-----------|----------|
| [SYSTEM_CARD.md](docs/SYSTEM_CARD.md) | Visão completa do sistema |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Detalhes do modelo LSTM |
| [OWASP_MAPPING.md](docs/OWASP_MAPPING.md) | 7 ameaças mapeadas |
| [RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md) | 7 cenários testados |
| [LGPD_PLAN.md](docs/LGPD_PLAN.md) | Conformidade LGPD + RIPD |
| [EDA_REPORT.md](docs/EDA_REPORT.md) | Análise exploratória |
| [TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | LSTM vs RF |
| [AB_TEST_REPORT.md](docs/AB_TEST_REPORT.md) | Benchmark 3 configs |

## Evolução Fase 4 → Fase 5

| Aspecto | Fase 4 | Fase 5 |
|---------|--------|--------|
| Hiperparâmetros | Manuais | Optuna (30 trials) |
| Features | Close | 25 indicadores técnicos |
| Modelo | LSTM apenas | LSTM + RF + Champion/Challenger |
| API | FastAPI básica | 8 endpoints + guardrails |
| Monitoramento | Prometheus básico | Drift + Grafana + Telemetria |
| Segurança | Nenhuma | OWASP + Red Team + LGPD |
| Avaliação | MAE/RMSE | RAGAS + LLM Judge + A/B Test |
| Tracking | Nenhum | MLflow + DVC |

## Equipe

| Nome | RM | Responsabilidade |
|------|----|--------------------|
| Giovanna Catelli | [RM] | Desenvolvimento completo (ML, LLM, API, segurança, docs) |

## Referências

- [RAGAS](https://arxiv.org/abs/2309.15217) — Automated Evaluation of RAG
- [ReAct](https://arxiv.org/abs/2210.03629) — Synergizing Reasoning and Acting
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)
- [Optuna](https://arxiv.org/abs/1907.10902) — Hyperparameter Optimization Framework
