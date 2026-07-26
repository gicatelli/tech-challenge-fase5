# Datathon Fase 05 — Agente Inteligente para Análise de Ações

**FIAP Pós-Tech MLET | Projeto Integrador (Fases 01–05)**

Sistema de IA com LLM local (Qwen2.5:3b), RAG (ChromaDB + embeddings locais) e Agente ReAct (4 tools) para análise e previsão de preços de ações brasileiras (PETR4.SA, VALE3.SA, ITUB4.SA).

---

## Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│              Etapa 1: Dados + Baseline (Fases 01-02)              │
│  yfinance → EDA → Feature Engineering (25 ind.) → LSTM+RF → MLflow│
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│              Etapa 2: LLM + Agente + RAG (Fases 03-05)            │
│  Qwen2.5:3b (Ollama) → Agente ReAct (4 tools) → FastAPI → CI/CD  │
│  ChromaDB + all-MiniLM-L6-v2 (embeddings locais)                  │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│           Etapa 3: Avaliação + Observabilidade (Fases 03-05)      │
│  RAGAS (4 métricas) → LLM Judge (4 critérios) → A/B Test          │
│  Prometheus + Grafana → Drift Detection (PSI) → Telemetria        │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│           Etapa 4: Segurança + Governança (Fases 04-05)           │
│  Guardrails (Input+Output) → OWASP 7/10 → Red Team 7/7 → LGPD    │
│  System Card → Model Card → Explicabilidade + Fairness             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Resultados

### Modelo LSTM (Champion)

| Métrica | LSTM | Random Forest | Melhor |
|---------|:----:|:-------------:|:------:|
| MAE | R$ 3.83 | R$ 3.86 | LSTM |
| RMSE | R$ 5.65 | R$ 6.20 | LSTM |
| R² | 0.295 | 0.150 | LSTM |
| Latência | 0.47ms | 0.11ms | RF |

### Avaliação RAG (RAGAS + LLM Judge)

| Métrica | Score |
|---------|:-----:|
| Faithfulness | 0.7005 |
| Answer Relevancy | 0.7248 |
| Context Precision | 0.6201 |
| Context Recall | 0.6156 |
| LLM Judge Overall | 3.62/5.0 |

### Segurança

| Item | Resultado |
|------|:---------:|
| Red Team | 7/7 cenários bloqueados |
| OWASP mapeado | 7/10 ameaças |
| Guardrails | Input + Output funcionais |
| Drift Detection | CRITICAL (PSI 1.57 — retrain recomendado) |

---

## Quick Start

### Pré-requisitos

- Python 3.11+
- Docker + Docker Compose (para infraestrutura local)
- Ollama (para LLM local gratuito)

### Instalação Local

```bash
# 1. Clonar repositório
git clone https://github.com/gicatelli/tech-challenge-fase5.git
cd tech-challenge-fase5
git checkout develop

# 2. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# 3. Instalar dependências (incluindo dev)
pip install -e ".[dev]"

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env conforme necessário

# 5. Instalar pre-commit hooks
pre-commit install

# 6. Instalar Ollama e baixar modelo
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

### Execução Completa (Local)

```bash
# Subir infraestrutura (MLflow, Prometheus, Grafana)
make docker-up

# Coletar dados (ou usar dados já versionados com DVC)
make data

# Treinar modelos (LSTM + Random Forest)
make train

# Ingerir documentos no RAG
make ingest

# Iniciar API
make serve

# Rodar testes
make test

# Avaliação completa (RAGAS + LLM Judge + A/B Test)
make evaluate

# Drift detection
make drift
```

### Execução no Google Colab (Recomendado para Avaliação)

O notebook `notebooks/colab_full_pipeline.ipynb` executa o pipeline completo end-to-end:

1. Abra no Google Colab
2. Execute todas as células em sequência
3. O notebook instala dependências, configura Ollama, e roda todo o pipeline
4. Resultados são salvos em `metrics/` e exibidos inline

**Link direto:** Abra `notebooks/colab_full_pipeline.ipynb` no GitHub e clique em "Open in Colab".

### Reprodução com DVC

```bash
# Baixar dados versionados
dvc pull

# Reproduzir pipeline completo
dvc repro
```

---

## Comandos Disponíveis (Makefile)

| Comando | Descrição |
|---------|-----------|
| `make install` | Instalar dependências de produção |
| `make dev` | Instalar dependências de desenvolvimento |
| `make lint` | Lint (ruff) + Type check (mypy) |
| `make test` | Rodar testes com coverage |
| `make train` | Treinar modelos (LSTM + RF) |
| `make serve` | Iniciar API FastAPI |
| `make data` | Coletar dados via yfinance |
| `make ingest` | Ingerir documentos no RAG |
| `make drift` | Executar drift detection |
| `make evaluate` | RAGAS + LLM Judge + A/B Test |
| `make docker-up` | Subir infraestrutura Docker |
| `make docker-down` | Derrubar infraestrutura Docker |
| `make clean` | Limpar arquivos temporários |

---

## Estrutura do Projeto

```
tech-challenge-fase5/
├── .github/
│   ├── workflows/ci.yml        # CI/CD: lint → mypy → bandit → pytest
│   ├── CODEOWNERS              # Ownership do código
│   └── pull_request_template.md
├── configs/
│   ├── model_config.yaml       # Hiperparâmetros dos modelos
│   ├── monitoring_config.yaml  # Thresholds de drift
│   ├── prometheus.yml          # Config Prometheus
│   └── grafana/                # Dashboards Grafana
├── data/
│   ├── raw/                    # Dados brutos (DVC tracked)
│   ├── processed/              # Features processadas
│   ├── golden_set/             # 20+ pares para avaliação RAGAS
│   └── knowledge_base/         # Documentos para RAG (17 docs)
├── docs/
│   ├── SYSTEM_CARD.md          # System Card completo
│   ├── MODEL_CARD.md           # Model Card do LSTM
│   ├── OWASP_MAPPING.md        # 7 ameaças mapeadas + mitigações
│   ├── RED_TEAM_REPORT.md      # 7 cenários adversariais
│   ├── LGPD_PLAN.md            # Conformidade LGPD + RIPD
│   ├── EXPLAINABILITY_FAIRNESS.md  # Explicabilidade + Fairness
│   ├── BUSINESS_METRICS.md     # Métricas técnicas → negócio
│   ├── EDA_REPORT.md           # Análise exploratória
│   ├── TRAINING_REPORT.md      # LSTM vs RF — comparação
│   └── AB_TEST_REPORT.md       # Benchmark 3 configs LLM
├── evaluation/
│   ├── ragas_eval.py           # RAGAS: 4 métricas
│   ├── llm_judge.py            # LLM-as-Judge: 4 critérios
│   └── ab_test_prompts.py      # A/B Test: 3 configurações
├── metrics/                    # Resultados gerados (JSON)
├── notebooks/
│   ├── 01_eda.ipynb            # EDA exploratória
│   └── colab_full_pipeline.ipynb  # Pipeline completo (Colab)
├── src/
│   ├── data_collection.py      # Coleta via yfinance (+ fallback sintético)
│   ├── features/
│   │   └── feature_engineering.py  # 25 indicadores técnicos + schema
│   ├── models/
│   │   ├── baseline.py         # MLP PyTorch + Random Forest
│   │   ├── hyperparameter_tuning.py  # Optuna (30 trials)
│   │   ├── train.py            # Pipeline de treino + MLflow
│   │   └── registry.py         # Model Registry + governança
│   ├── agent/
│   │   ├── react_agent.py      # Agente ReAct (Ollama/Gemini/OpenAI)
│   │   ├── tools.py            # 4 tools: prever, risco, histórico, busca
│   │   └── rag_pipeline.py     # RAG: ChromaDB + embeddings locais
│   ├── serving/
│   │   ├── app.py              # FastAPI (8 endpoints)
│   │   └── Dockerfile          # Container de produção
│   ├── monitoring/
│   │   ├── drift.py            # Drift detection (PSI + Evidently)
│   │   ├── metrics.py          # Prometheus custom metrics
│   │   └── telemetry.py        # Tracing (MLflow + Langfuse)
│   └── security/
│       ├── guardrails.py       # Input/Output guardrails
│       └── pii_detection.py    # Presidio PII detection
├── tests/                      # ~160 testes (pytest, coverage >60%)
│   ├── conftest.py             # Fixtures compartilhados
│   ├── test_features.py        # Feature engineering
│   ├── test_models.py          # Baseline MLP + RF
│   ├── test_train.py           # Pipeline de treinamento
│   ├── test_agent.py           # Tools do agente
│   ├── test_react_agent.py     # Agente ReAct
│   ├── test_rag_pipeline.py    # RAG pipeline
│   ├── test_api.py             # Endpoints FastAPI
│   ├── test_drift.py           # Drift detection
│   ├── test_telemetry.py       # Telemetria
│   ├── test_data_collection.py # Coleta de dados
│   ├── test_registry.py        # Model Registry
│   └── test_guardrails.py      # Segurança
├── docker-compose.yml          # MLflow + Prometheus + Grafana
├── dvc.yaml                    # Pipeline DVC
├── pyproject.toml              # Dependências + configs (ruff, mypy, pytest)
├── Makefile                    # Atalhos de execução
├── .pre-commit-config.yaml     # Hooks de qualidade
└── .env.example                # Template de variáveis
```

---

## Documentação Completa

| Documento | Conteúdo | Requisito |
|-----------|----------|:---------:|
| [SYSTEM_CARD.md](docs/SYSTEM_CARD.md) | Visão completa do sistema | Etapa 4 |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Detalhes do LSTM + métricas | Etapa 4 |
| [OWASP_MAPPING.md](docs/OWASP_MAPPING.md) | 7 ameaças mapeadas + mitigações | Etapa 4 |
| [RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md) | 7 cenários adversariais testados | Etapa 4 |
| [LGPD_PLAN.md](docs/LGPD_PLAN.md) | Conformidade LGPD + RIPD | Etapa 4 |
| [EXPLAINABILITY_FAIRNESS.md](docs/EXPLAINABILITY_FAIRNESS.md) | Explicabilidade + Fairness | Etapa 4 |
| [BUSINESS_METRICS.md](docs/BUSINESS_METRICS.md) | Métricas técnicas → impacto R$ | Empresa |
| [EDA_REPORT.md](docs/EDA_REPORT.md) | Análise exploratória | Etapa 1 |
| [TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | Comparação LSTM vs RF | Etapa 1 |
| [AB_TEST_REPORT.md](docs/AB_TEST_REPORT.md) | Benchmark 3 configs de LLM | Etapa 3 |

---

## Evolução Fase 4 → Fase 5

| Aspecto | Fase 4 | Fase 5 |
|---------|--------|--------|
| Hiperparâmetros | Manuais | Optuna (30 trials, busca bayesiana) |
| Features | 5 (Close, SMA, RSI) | 25 indicadores técnicos |
| Modelo | LSTM apenas | LSTM + RF + Champion/Challenger |
| LLM | Nenhum | Qwen2.5:3b local (gratuito) |
| RAG | Nenhum | ChromaDB + 17 documentos + embeddings locais |
| Agente | Nenhum | ReAct com 4 tools customizadas |
| API | FastAPI básica (2 endpoints) | 8 endpoints + guardrails + health check |
| Monitoramento | Prometheus básico | Drift (PSI) + Grafana + Telemetria |
| Segurança | Nenhuma | OWASP 7/10 + Red Team 7/7 + LGPD |
| Avaliação | MAE/RMSE | RAGAS + LLM Judge + A/B Test |
| Tracking | Nenhum | MLflow (experimentos + registry) + DVC |
| Testes | Nenhum | ~160 testes, coverage >60% |
| CI/CD | Nenhum | GitHub Actions (lint + mypy + bandit + pytest) |
| Documentação | README básico | System Card + Model Card + 8 docs |

---

## Tecnologias

| Categoria | Tecnologia |
|-----------|-----------|
| ML Framework | PyTorch (LSTM) + Scikit-Learn (RF) |
| Otimização | Optuna |
| LLM | Qwen2.5:3b (Ollama, local) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers, local) |
| Vector Store | ChromaDB |
| Agente | LangChain (ReAct) |
| API | FastAPI |
| Tracking | MLflow |
| Data Version | DVC |
| Monitoramento | Prometheus + Grafana |
| Drift | Evidently + PSI customizado |
| Segurança | Presidio (PII) + regex guardrails |
| CI/CD | GitHub Actions |
| Containerização | Docker + Docker Compose |
| Testes | pytest + coverage |
| Qualidade | ruff + mypy + bandit + pre-commit |

---

## API Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/metrics/summary` | Métricas do sistema |
| GET | `/tools` | Tools disponíveis |
| POST | `/predict` | Previsão de preço |
| POST | `/analyze` | Análise histórica |
| POST | `/risk` | Cálculo de risco |
| POST | `/query` | Agente ReAct completo |
| POST | `/rag/query` | Query RAG direta |

---

## Equipe

| Nome | RM | Responsabilidade |
|------|----|--------------------|
| Giovanna Catelli | — | Desenvolvimento completo (ML, LLM, API, segurança, docs) |

---

## Referências

- Es, S. et al. (2024). [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217)
- Yao, S. et al. (2023). [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- Akiba, T. et al. (2019). [Optuna: Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)
- [OWASP Top 10 for LLM Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.
- Brasil. Lei nº 13.709/2018 (LGPD).
