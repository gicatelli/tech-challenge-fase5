# Relatório de Treinamento — LSTM vs Random Forest

## Resumo

Pipeline de treinamento para previsão de preços de PETR4.SA comparando:
- **LSTM** (PyTorch) — modelo sequencial para séries temporais
- **Random Forest** (Scikit-Learn) — baseline interpretável

## Arquitetura dos Modelos

### LSTM (Champion esperado)
- Input: Sequências de 60 dias × N features (indicadores técnicos)
- Arquitetura: LSTM multicamada → Dense → ReLU → Dropout → Output
- Hiperparâmetros: Otimizados via Optuna (busca bayesiana, 30 trials)
- Treinamento: Adam + ReduceLROnPlateau + Early Stopping + Gradient Clipping

### Random Forest (Baseline)
- Input: Último timestep da sequência (vetor de features)
- Hiperparâmetros: `n_estimators=100, max_depth=10`
- Vantagens: Rápido, interpretável, feature importance nativa

## Métricas de Comparação

| Métrica | LSTM | Random Forest | Melhor |
|---------|------|---------------|--------|
| MAE (R$) | - | - | - |
| RMSE (R$) | - | - | - |
| MAPE (%) | - | - | - |
| R² | - | - | - |
| Latência (ms/sample) | - | - | - |

> Métricas serão preenchidas após execução do pipeline com `python -m src.models.train`

## Critério de Seleção do Champion

O champion é selecionado automaticamente pelo **menor RMSE** no conjunto de teste.

**Justificativa**: RMSE penaliza erros grandes mais que MAE, o que é desejável em
previsão de preços onde um erro extremo pode causar decisão de investimento incorreta.

## Evolução em relação à Fase 4

| Aspecto | Fase 4 | Fase 5 |
|---------|--------|--------|
| Hiperparâmetros | Manuais (hardcoded) | Optuna (30 trials, busca bayesiana) |
| Features | Apenas preço Close | 25+ indicadores técnicos |
| Comparação | Nenhuma (só LSTM) | LSTM vs RF + champion selection |
| Tracking | Sem tracking | MLflow (params, metrics, artifacts) |
| Reprodutibilidade | Manual | DVC pipeline + configs YAML |

## Como Executar

```bash
# Subir MLflow
docker-compose up -d mlflow

# Treinar ambos modelos
python -m src.models.train

# Ou via Makefile
make train
```

## Artefatos Gerados

- `metrics/train_metrics.json` — Comparação LSTM vs RF
- MLflow runs com modelos serializados
- `configs/model_config.yaml` — Hiperparâmetros otimizados
