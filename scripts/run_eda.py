"""Executa a EDA e gera imagens para docs/img/."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Config
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 11

DATA_DIR = Path("data/raw")
IMG_DIR = Path("docs/img")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Carregar dados
petr4 = pd.read_csv(DATA_DIR / "PETR4_SA_historico.csv", index_col=0, parse_dates=True)
vale3 = pd.read_csv(DATA_DIR / "VALE3_SA_historico.csv", index_col=0, parse_dates=True)
itub4 = pd.read_csv(DATA_DIR / "ITUB4_SA_historico.csv", index_col=0, parse_dates=True)

print(f"PETR4: {petr4.shape}")
print(f"VALE3: {vale3.shape}")
print(f"ITUB4: {itub4.shape}")

# === 1. Série Temporal com Médias Móveis ===
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
for ax, (name, df) in zip(axes, [("PETR4.SA", petr4), ("VALE3.SA", vale3), ("ITUB4.SA", itub4)]):
    ax.plot(df.index, df["Close"], alpha=0.7, linewidth=0.8, label="Close")
    ax.plot(df.index, df["Close"].rolling(30).mean(), linewidth=1.5, label="SMA 30")
    ax.plot(df.index, df["Close"].rolling(90).mean(), linewidth=1.5, label="SMA 90")
    ax.set_title(f"{name} — Preço de Fechamento + Médias Móveis")
    ax.set_ylabel("Preço (R$)")
    ax.legend(loc="upper left")
axes[-1].set_xlabel("Data")
plt.tight_layout()
plt.savefig(IMG_DIR / "serie_temporal_completa.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ serie_temporal_completa.png")

# === 2. Retornos Diários ===
petr4["returns"] = np.log(petr4["Close"] / petr4["Close"].shift(1))
vale3["returns"] = np.log(vale3["Close"] / vale3["Close"].shift(1))
itub4["returns"] = np.log(itub4["Close"] / itub4["Close"].shift(1))

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, (name, df) in zip(axes, [("PETR4", petr4), ("VALE3", vale3), ("ITUB4", itub4)]):
    ax.plot(df.index, df["returns"], alpha=0.6, linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(f"{name} — Retornos Diários (log)")
    ax.set_ylabel("Retorno")
axes[-1].set_xlabel("Data")
plt.tight_layout()
plt.savefig(IMG_DIR / "retornos_diarios.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ retornos_diarios.png")

# === 3. Volatilidade Rolling ===
fig, ax = plt.subplots(figsize=(14, 6))
for name, df in [("PETR4", petr4), ("VALE3", vale3), ("ITUB4", itub4)]:
    vol = df["returns"].rolling(30).std() * np.sqrt(252)
    ax.plot(df.index, vol, label=f"{name} (vol 30d)", linewidth=1)
ax.set_title("Volatilidade Rolling 30 dias (Anualizada)")
ax.set_ylabel("Volatilidade")
ax.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "volatilidade_rolling.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ volatilidade_rolling.png")

# === 4. Drawdown ===
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, (name, df) in zip(axes, [("PETR4", petr4), ("VALE3", vale3), ("ITUB4", itub4)]):
    cummax = df["Close"].cummax()
    drawdown = (df["Close"] - cummax) / cummax
    ax.fill_between(df.index, drawdown, 0, alpha=0.4, color="red")
    ax.set_title(f"{name} — Drawdown")
    ax.set_ylabel("Drawdown (%)")
axes[-1].set_xlabel("Data")
plt.tight_layout()
plt.savefig(IMG_DIR / "drawdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ drawdown.png")

# === 5. Sazonalidade ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
petr4["month"] = petr4.index.month
monthly_returns = petr4.groupby("month")["returns"].mean()
axes[0].bar(monthly_returns.index, monthly_returns.values, color="steelblue", alpha=0.7)
axes[0].axhline(y=0, color="black", linewidth=0.5)
axes[0].set_title("PETR4 — Retorno Médio por Mês")
axes[0].set_xlabel("Mês")
axes[0].set_xticks(range(1, 13))

petr4["weekday"] = petr4.index.dayofweek
weekday_returns = petr4.groupby("weekday")["returns"].mean()
days = ["Seg", "Ter", "Qua", "Qui", "Sex"]
axes[1].bar(range(5), weekday_returns.values[:5], color="coral", alpha=0.7)
axes[1].axhline(y=0, color="black", linewidth=0.5)
axes[1].set_title("PETR4 — Retorno Médio por Dia da Semana")
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(days)
plt.tight_layout()
plt.savefig(IMG_DIR / "sazonalidade.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ sazonalidade.png")

# === 6. Correlação ===
returns_df = pd.DataFrame({
    "PETR4": petr4["returns"],
    "VALE3": vale3["returns"],
    "ITUB4": itub4["returns"],
}).dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
corr = returns_df.corr()
sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=axes[0], vmin=-1, vmax=1, fmt=".3f")
axes[0].set_title("Correlação dos Retornos Diários")

rolling_corr = returns_df["PETR4"].rolling(60).corr(returns_df["VALE3"])
axes[1].plot(returns_df.index, rolling_corr, linewidth=1, label="PETR4 vs VALE3")
rolling_corr2 = returns_df["PETR4"].rolling(60).corr(returns_df["ITUB4"])
axes[1].plot(returns_df.index, rolling_corr2, linewidth=1, label="PETR4 vs ITUB4")
axes[1].axhline(y=0, color="black", linewidth=0.5, linestyle="--")
axes[1].set_title("Correlação Rolling 60 dias")
axes[1].legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "correlacao.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ correlacao.png")

# === 7. Distribuição dos Retornos ===
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, df) in zip(axes, [("PETR4", petr4), ("VALE3", vale3), ("ITUB4", itub4)]):
    r = df["returns"].dropna()
    ax.hist(r, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(x=r.mean(), color="red", linestyle="--", label=f"μ={r.mean():.4f}")
    ax.set_title(f"{name} — Distribuição Retornos")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(IMG_DIR / "distribuicao_retornos.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ distribuicao_retornos.png")

# === INSIGHTS ===
print("\n" + "=" * 60)
print("  INSIGHTS DE NEGÓCIO")
print("=" * 60)

valoriz = (petr4["Close"].iloc[-1] / petr4["Close"].iloc[0] - 1) * 100
print(f"\n1. Valorização PETR4 (2018-2026): {valoriz:.0f}%")

vol_30d = petr4["returns"].rolling(30).std() * np.sqrt(252)
print(f"2. Volatilidade média anualizada PETR4: {vol_30d.mean():.1%}")

print(f"3. Correlação PETR4-VALE3: {corr.loc['PETR4','VALE3']:.3f}")
print(f"   Correlação PETR4-ITUB4: {corr.loc['PETR4','ITUB4']:.3f}")

vol_by_month = petr4.groupby("month")["returns"].std()
print(f"4. Mês mais volátil: {vol_by_month.idxmax()} (std={vol_by_month.max():.4f})")

kurt = petr4["returns"].dropna().kurtosis()
print(f"5. Curtose: {kurt:.2f} (caudas pesadas — eventos extremos mais frequentes)")

dd = (petr4["Close"] - petr4["Close"].cummax()) / petr4["Close"].cummax()
print(f"   Drawdown máximo PETR4: {dd.min():.1%}")

print(f"\n✅ EDA completa! Imagens salvas em {IMG_DIR.resolve()}")
