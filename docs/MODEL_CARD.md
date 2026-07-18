# Model Card — LSTM Predictor (PETR4.SA)

## Informações Gerais

| Campo | Valor |
|-------|-------|
| **Nome** | lstm-petr4-predictor |
| **Versão** | 1.0.0 |
| **Tipo** | Regressão (previsão de preço) |
| **Framework** | PyTorch |
| **Ativo** | PETR4.SA (Petrobras) |
| **Proprietário** | Giovanna Catelli |
| **Data de treino** | Julho 2026 |
| **Otimização** | Optuna (busca bayesiana, 30 trials) |

## Arquitetura

```
Input (60 dias × 25 features)
    ↓
LSTM (hidden_size_1=128, num_layers=2, dropout=0.2)
    ↓
Dense (hidden_size_2=64)
    ↓
ReLU + Dropout
    ↓
Output (1 valor: preço próximo dia)
```

## Dados de Treinamento

| Item | Valor |
|------|-------|
| Período | 2018-01-02 a 2026-05-19 |
| Registros totais | 2081 |
| Split | 80% treino / 20% teste (temporal) |
| Features | 25 indicadores técnicos |
| Target | Preço de fechamento (Close) normalizado |

## Features (top 10 por importância)

| # | Feature | Importância |
|---|---------|:-----------:|
| 1 | log_return_1d | 24.07% |
| 2 | log_return_5d | 21.75% |
| 3 | sma_7_ratio | 11.11% |
| 4 | volume_norm | 10.22% |
| 5 | rsi_14 | 9.54% |
| 6 | sma_90_ratio | 8.90% |
| 7 | sma_30_ratio | 6.16% |
| 8 | daily_range | 4.45% |
| 9 | volatility_30 | 2.60% |
| 10 | bb_width | 1.21% |

## Métricas (Fase 4 — baseline manual)

| Métrica | Valor Fase 4 | Esperado Fase 5 (Optuna) |
|---------|:---:|:---:|
| MAE | R$ 1.74 | < R$ 1.50 |
| RMSE | R$ 2.27 | < R$ 2.00 |
| MAPE | 4.87% | < 4.0% |

## Hiperparâmetros

| Parâmetro | Fase 4 (manual) | Fase 5 (Optuna) |
|-----------|:---:|:---:|
| hidden_size_1 | 256 | Otimizado (32-256) |
| hidden_size_2 | 128 | Otimizado (16-128) |
| num_layers | 3 | Otimizado (1-3) |
| dropout | 0.2 | Otimizado (0.1-0.5) |
| learning_rate | 0.0005 | Otimizado (1e-4 a 1e-2) |
| batch_size | 32 | Otimizado (16-128) |
| sequence_length | 90 | 60 |

## Limitações

1. **Não prevê eventos imprevisíveis** (guerras, pandemias, escândalos)
2. **Performance degrada em mudanças de regime** (bull → bear)
3. **Requer retreinamento periódico** para adaptar a novos padrões
4. **Latência computacional** maior que modelos lineares
5. **Sem dados fundamentalistas** (balanço, DRE, notícias)

## Considerações Éticas

- Previsões **NÃO são recomendações de investimento**
- Disclaimer obrigatório em todas as respostas de previsão
- Modelo pode estar errado — usuário deve fazer própria análise
- Análise de fairness mostra viés NEUTRO (51% up / 49% down)

## Reprodutibilidade

| Item | Valor |
|------|-------|
| Git SHA | Verificar via `git log --oneline -1` |
| DVC hash | `data/raw.dvc` |
| MLflow run_id | Gerado ao executar `python -m src.models.train` |
| Random seed | 42 |
| Python | 3.11 |
| PyTorch | 2.2+ |

## Monitoramento Pós-Deploy

- Drift detection via PSI a cada 6 horas
- Retrain trigger: PSI > 0.2 em qualquer feature
- Champion-Challenger: só promover se RMSE melhorar 0.5%+
- Approval gate: human-in-the-loop antes de produção
