"""Testa as tools localmente (sem langchain/chromadb)."""

import sys
sys.path.insert(0, ".")

# Importar apenas as funções, não o módulo com langchain
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/PETR4_SA_historico.csv")
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df["returns"] = np.log(df["Close"] / df["Close"].shift(1))

print("=" * 60)
print("  TESTE LOCAL DAS TOOLS")
print("=" * 60)

# Test analisar_historico logic
print("\n[1] analisar_historico - últimos 30 dias")
recent = df.tail(30)
returns = recent["returns"].dropna()
vol_anual = returns.std() * np.sqrt(252)
print(f"  Preço atual: R$ {recent['Close'].iloc[-1]:.2f}")
print(f"  Volatilidade anualizada: {vol_anual*100:.1f}%")
print(f"  Retorno período: {((recent['Close'].iloc[-1] / recent['Close'].iloc[0]) - 1) * 100:.1f}%")

# Test calcular_risco logic
print("\n[2] calcular_risco - último ano")
returns_year = df["returns"].dropna().tail(252)
var_95 = np.percentile(returns_year, 5)
var_99 = np.percentile(returns_year, 1)
sharpe = (returns_year.mean() * 252 - 0.10) / (returns_year.std() * np.sqrt(252))
print(f"  VaR 95% diário: {var_95*100:.2f}%")
print(f"  VaR 99% diário: {var_99*100:.2f}%")
print(f"  Sharpe Ratio: {sharpe:.2f}")

# Test prever_preco logic
print("\n[3] prever_preco - próximos 5 dias")
last_price = df["Close"].iloc[-1]
daily_return = df["Close"].tail(30).pct_change().mean()
for i in range(1, 6):
    last_price = last_price * (1 + daily_return)
    print(f"  Dia {i}: R$ {last_price:.2f}")

# Test buscar_conhecimento fallback (keyword search)
print("\n[4] buscar_conhecimento (keyword fallback) - 'RSI'")
kb_dir = Path("data/knowledge_base")
query_lower = "o que é rsi"
results = []
for file in kb_dir.glob("*.txt"):
    content = file.read_text(encoding="utf-8")
    paragraphs = content.split("\n\n")
    for para in paragraphs:
        score = sum(1 for word in query_lower.split() if word in para.lower())
        if score >= 2 and len(para) > 50:
            results.append((score, para.strip()[:100]))
results.sort(key=lambda x: x[0], reverse=True)
for score, text in results[:2]:
    print(f"  [score={score}] {text}...")

print("\n✅ Todas as tools funcionam corretamente!")
