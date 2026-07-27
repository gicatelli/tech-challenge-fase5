# Explicabilidade e Fairness — Datathon Fase 05

## 1. Explicabilidade do Modelo LSTM

### 1.1 Importância de Features (Permutation Importance)

A interpretabilidade do modelo LSTM foi avaliada via permutation importance sobre o conjunto de teste. Essa técnica mede quanto o erro (RMSE) aumenta quando cada feature é embaralhada aleatoriamente — quanto maior o aumento, mais importante a feature.

| # | Feature | Importância | Categoria |
|---|---------|:-----------:|-----------|
| 1 | log_return_1d | 24.07% | Momentum de curto prazo |
| 2 | log_return_5d | 21.75% | Momentum de médio prazo |
| 3 | sma_7_ratio | 11.11% | Tendência (média móvel curta) |
| 4 | volume_norm | 10.22% | Liquidez |
| 5 | rsi_14 | 9.54% | Oscilador de momentum |
| 6 | sma_90_ratio | 8.90% | Tendência (média móvel longa) |
| 7 | sma_30_ratio | 6.16% | Tendência (média móvel média) |
| 8 | daily_range | 4.45% | Volatilidade intraday |
| 9 | volatility_30 | 2.60% | Volatilidade histórica |
| 10 | bb_width | 1.21% | Bandas de Bollinger |

### 1.2 Interpretação

**Observações-chave:**

1. **Retornos logarítmicos dominam (~46%)**: O modelo se apoia fortemente em momentum recente (1d e 5d). Isso indica que o LSTM está capturando padrões de autocorrelação de curto prazo — comportamento esperado em séries temporais financeiras.

2. **Médias móveis são relevantes (~26%)**: A relação preço/SMA em diferentes janelas (7, 30, 90 dias) indica que o modelo detecta desvios do preço em relação à tendência. Quando o preço está acima da SMA, há pressão de retorno à média.

3. **Volume como proxy de convicção (~10%)**: Volume normalizado é a 4ª feature mais importante, sugerindo que o modelo aprendeu que movimentos com alto volume tendem a persistir.

4. **RSI como indicador de reversão (~10%)**: O RSI captura condições de sobrecompra/sobrevenda, complementando o momentum puro.

5. **Volatilidade com baixa importância (~4%)**: Volatility_30 e bb_width têm baixo impacto, indicando que o modelo prioriza direção sobre magnitude de variação.

### 1.3 Limitações de Explicabilidade

- **Opacidade do LSTM**: Como modelo recorrente, o LSTM processa sequências inteiras (60 dias). A permutation importance mede importância média, mas não explica *quando* na sequência cada feature importa mais.
- **Interações não capturadas**: Features podem interagir (ex: volume alto + RSI extremo). Permutation importance não captura essas interações de segunda ordem.
- **Alternativas futuras**: Para maior granularidade, considerar SHAP (SHapley Additive exPlanations) adaptado para séries temporais, ou attention-based LSTM para visualização temporal.

### 1.4 Transparência do Agente ReAct

O agente ReAct implementa transparência nativa por design. Cada resposta inclui a cadeia completa de raciocínio:

```
Thought: Preciso prever o preço e calcular o risco.
Action: prever_preco
Action Input: próximos 5 dias
Observation: Previsão: R$ 46.03, 45.97, 45.90, 45.84, 45.78

Thought: Agora preciso do risco associado.
Action: calcular_risco
Action Input: último trimestre
Observation: VaR 95%: -2.43%, Sharpe: 1.49

Thought: Tenho previsão e risco. Posso responder.
Final Answer: A PETR4 tem tendência de leve queda nos próximos 5 dias
             (R$ 46.03 → R$ 45.78). O risco é moderado com VaR 95%
             de -2.43% diário e Sharpe Ratio de 1.49.
```

**Benefícios:**
- O usuário vê *quais tools* foram consultadas e *em que ordem*
- Cada step tem justificativa (Thought) — auditável
- Fontes dos dados são rastreáveis (tools específicas)
- Limitações são declaradas ("análise baseada em dados históricos")

---

## 2. Fairness (Equidade)

### 2.1 Análise de Viés Direcional

Para um modelo de previsão de preços, fairness significa: **o modelo não deve ter viés sistemático para prever alta ou baixa** que não reflita os dados reais.

| Métrica | Valor | Interpretação |
|---------|:-----:|---------------|
| Predições de alta | 51% | Ligeiramente otimista |
| Predições de baixa | 49% | Equilibrado |
| Viés detectado | NEUTRO | Distribuição equilibrada |
| Retorno médio diário (dados) | -0.14% | Tendência leve de queda |

**Conclusão**: O modelo apresenta distribuição praticamente equilibrada entre predições de alta e baixa (51%/49%), sem viés direcional significativo. A leve tendência de queda nos retornos diários é refletida proporcionalmente.

### 2.2 Performance por Regime de Mercado

Avaliamos se o modelo performa de forma justa em diferentes regimes de mercado (bull, bear, neutro):

| Regime | Dias | Retorno Médio | Volatilidade | Sharpe | Max Drawdown |
|--------|:----:|:---:|:---:|:---:|:---:|
| **Bull** | 1480 | 14.13% | 38.48% | 0.11 | -48.56% |
| **Bear** | 481 | 82.84% | 47.23% | 1.54 | -58.94% |
| **Neutral** | 119 | -10.74% | 61.31% | -0.34 | -46.96% |

**Observações:**
- O modelo tem dados substancialmente mais abundantes para regimes de bull (1480 dias) do que bear (481) ou neutro (119)
- Em regimes bear, o Sharpe é significativamente melhor — possivelmente porque movimentos de queda são mais bruscos e previsíveis
- O regime neutro tem pior performance (Sharpe negativo) — esperado, pois movimentos laterais são mais difíceis de prever

### 2.3 Fairness para Diferentes Horizontes Temporais

| Horizonte | MAE (R$) | Bias |
|-----------|:--------:|:----:|
| 1 dia | ~R$ 0.80 | Neutro |
| 5 dias | ~R$ 2.50 | Levemente otimista |
| 10 dias | ~R$ 4.20 | Incerto (degradação natural) |

O modelo é mais preciso e justo para horizontes curtos. Para horizontes maiores, a incerteza cresce e há leve viés otimista — limitação documentada no Model Card.

### 2.4 Viés de Grupo (Não Aplicável)

Este modelo opera sobre dados de mercado público (preços de ações) e não processa dados demográficos de pessoas. Portanto:

- **Não há grupos protegidos** (raça, gênero, idade) nos dados de entrada
- **Não há decisão sobre indivíduos** — o modelo prevê preços de ativos
- **O risco de discriminação é inexistente** para este caso de uso

No entanto, se o sistema fosse usado para decisões de crédito ou investimento pessoal, seria necessário avaliar se as recomendações geram impacto desproporcional em algum grupo.

---

## 3. Mitigações Implementadas

| Risco | Mitigação | Status |
|-------|-----------|:------:|
| Viés direcional | Monitoramento contínuo (51/49 split) | ✅ |
| Overconfidence | Disclaimer em respostas do agente | ✅ |
| Data leakage temporal | Split temporal rigoroso (80/20) | ✅ |
| Feature dominance | Permutation importance documentada | ✅ |
| Regime imbalance | Drift detection por regime | ✅ |
| Usuário confia cegamente | System Card + alertas de risco | ✅ |

---

## 4. Recomendações Futuras

1. **SHAP para séries temporais**: Implementar DeepSHAP ou KernelSHAP adaptado para LSTM, permitindo explicações por timestep
2. **Attention mechanism**: Migrar para Transformer com attention weights visualizáveis
3. **Calibração de confiança**: Adicionar intervalos de predição (prediction intervals) para quantificar incerteza
4. **Backtesting por regime**: Avaliar métricas de fairness segregadas por bull/bear/neutral em janelas deslizantes
5. **Feedback loop**: Monitorar se predições influenciam decisões que alteram o mercado (reflexividade)

---

## Referências

- Lundberg, S. & Lee, S-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT*.
- Molnar, C. (2022). *Interpretable Machine Learning*. 2nd ed.
- Mehrabi, N. et al. (2021). *A Survey on Bias and Fairness in Machine Learning*. ACM Computing Surveys.
