# Model Card — Datathon Fase 05

## Informações do Modelo

| Campo | Valor |
|-------|-------|
| Nome | [Nome do modelo] |
| Versão | 1.0.0 |
| Tipo | Classification |
| Framework | Scikit-Learn / PyTorch |
| Owner | Grupo XX |
| Risk Level | Medium |
| Fairness Checked | ☐ Sim / ☐ Não |

## Descrição

### Objetivo
[Descrever o que o modelo faz e qual problema resolve]

### Arquitetura
- **Baseline 1**: Random Forest (n_estimators=100, max_depth=10)
- **Baseline 2**: MLP PyTorch ([64, 32, 16], dropout=0.3)

## Dados de Treinamento

| Campo | Valor |
|-------|-------|
| Fonte | [Empresa convidada] |
| Volume | [X registros] |
| Features | [N features] |
| Target | [Nome da variável target] |
| Split | 80% treino / 20% teste |
| Versionamento | DVC hash: [hash] |

### Distribuição do Target
- Classe 0: X% (N registros)
- Classe 1: X% (N registros)

## Métricas de Performance

### Random Forest
| Métrica | Treino | Teste |
|---------|--------|-------|
| AUC | X.XX | X.XX |
| F1 | X.XX | X.XX |
| Precision | X.XX | X.XX |
| Recall | X.XX | X.XX |

### MLP PyTorch
| Métrica | Treino | Teste |
|---------|--------|-------|
| AUC | X.XX | X.XX |
| F1 | X.XX | X.XX |
| Precision | X.XX | X.XX |
| Recall | X.XX | X.XX |

## Limitações

1. [Limitação 1 — ex: performance degrada com dados fora da distribuição]
2. [Limitação 2 — ex: não captura padrões temporais]
3. [Limitação 3]

## Considerações Éticas

### Viés e Fairness
- [Análise de viés realizada]
- [Grupos protegidos considerados]

### Uso Responsável
- O modelo NÃO deve ser usado como única fonte de decisão
- Requer supervisão humana para decisões de alto impacto
- Não deve ser aplicado fora do domínio de treinamento

## Reprodutibilidade

```bash
# Reproduzir treinamento
make data
make train

# Verificar métricas
mlflow ui --port 5000
```

## Versionamento

| Versão | Data | Mudança | Métricas |
|--------|------|---------|----------|
| 1.0.0 | YYYY-MM-DD | Baseline inicial | AUC=X.XX |

## Referências

- MLflow Run ID: [run_id]
- Git SHA: [commit_hash]
- DVC Data Version: [dvc_hash]
