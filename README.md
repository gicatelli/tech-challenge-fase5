# Datathon Fase 05 — FIAP Pós-Tech MLET

## 📋 Visão Geral

Projeto integrador (Fases 01–05) do Datathon da Pós-Tech em Machine Learning Engineering.
Sistema de IA com LLM, RAG e Agente ReAct para o domínio financeiro.

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    Etapa 1: Dados + Baseline                 │
│  [DVC] → [EDA] → [Feature Eng.] → [Baseline] → [MLflow]    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Etapa 2: LLM + Agente + RAG                 │
│  [LLM Serving] → [Agente ReAct] → [RAG] → [FastAPI] → [CI] │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Etapa 3: Avaliação + Observabilidade             │
│  [Golden Set] → [RAGAS] → [LLM Judge] → [Prometheus/Grafana]│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Etapa 4: Segurança + Governança                 │
│  [Guardrails] → [OWASP] → [Red Team] → [LGPD] → [System Card]│
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Pré-requisitos
- Python 3.11+
- Docker + Docker Compose
- Git

### Instalação

```bash
# Clonar repositório
git clone <repo-url>
cd datathon-fase05

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -e ".[dev]"

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas chaves

# Instalar pre-commit hooks
pre-commit install
```

### Executar

```bash
# Subir infraestrutura (MLflow, Prometheus, Grafana)
docker-compose up -d

# Treinar modelo baseline
make train

# Iniciar API
make serve

# Rodar testes
make test

# Lint
make lint
```

## 📁 Estrutura do Projeto

```
datathon-fase05/
├── .github/workflows/ci.yml    # CI/CD: lint + test + build
├── data/
│   ├── raw/                    # Dados brutos (DVC)
│   ├── processed/              # Dados processados
│   └── golden_set/             # ≥ 20 pares para avaliação
├── src/
│   ├── features/               # Feature engineering
│   ├── models/                 # Baseline (sklearn + PyTorch)
│   ├── agent/                  # Agente ReAct + RAG
│   ├── serving/                # FastAPI + Dockerfile
│   ├── monitoring/             # Drift detection + métricas
│   └── security/               # Guardrails + PII
├── tests/                      # Testes (pytest)
├── evaluation/                 # RAGAS + LLM Judge
├── docs/                       # Model Card, System Card, LGPD
├── configs/                    # Configurações YAML
├── docker-compose.yml          # Orquestração
├── dvc.yaml                    # Pipeline DVC
├── pyproject.toml              # Dependências
└── Makefile                    # Atalhos
```

## 📊 Métricas

### Baseline
| Modelo | AUC | F1 | Precision | Recall |
|--------|-----|----|-----------| -------|
| Random Forest | - | - | - | - |
| MLP PyTorch | - | - | - | - |

### RAG (RAGAS)
| Métrica | Score |
|---------|-------|
| Faithfulness | - |
| Answer Relevancy | - |
| Context Precision | - |
| Context Recall | - |

## 🔒 Segurança

- Guardrails de input (prompt injection, exfiltração)
- Guardrails de output (PII removal via Presidio)
- OWASP Top 10 mapeado (≥ 5 ameaças)
- Red Teaming (≥ 5 cenários)
- Conformidade LGPD

## 👥 Equipe

| Nome | RM | Responsabilidade |
|------|----|--------------------|
| [Nome 1] | [RM] | [Responsabilidade] |
| [Nome 2] | [RM] | [Responsabilidade] |
| [Nome 3] | [RM] | [Responsabilidade] |

## 📚 Referências

- [RAGAS](https://arxiv.org/abs/2309.15217) — Automated Evaluation of RAG
- [ReAct](https://arxiv.org/abs/2210.03629) — Synergizing Reasoning and Acting
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)
