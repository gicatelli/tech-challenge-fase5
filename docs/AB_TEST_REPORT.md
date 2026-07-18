# Benchmark de Configurações — A/B Test

## Configurações Testadas

| Config | Temperature | Top-K | Descrição |
|--------|-------------|-------|-----------|
| A_conservador | 0.0 | 3 | Conservador — respostas determinísticas, poucos contextos |
| B_balanceado | 0.3 | 5 | Balanceado — leve variação, mais contextos |
| C_exploratorio | 0.7 | 10 | Exploratório — alta variação, máximo de contextos |

## Resultados

| Config | Latência Média (ms) | P95 (ms) | Tamanho Médio Resposta |
|--------|--------------------:|--------:|-----------------------:|
| A_conservador | 63.75 | 569.98 | 252 chars |
| B_balanceado | 7.47 | 9.9 | 252 chars |
| C_exploratorio | 8.03 | 11.07 | 252 chars |

## Recomendação

**Config recomendada para produção: B_balanceado**

Critérios: menor latência com qualidade aceitável.

Para o Datathon, `B_balanceado` oferece o melhor trade-off entre qualidade (variação moderada) e performance.
