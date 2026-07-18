"""Gera metadata da coleta de dados."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

raw_dir = Path("data/raw")
df = pd.read_csv(raw_dir / "PETR4_SA_historico.csv", index_col=0, parse_dates=True)

metadata = {
    "collected_at": datetime.now().isoformat(),
    "symbols": ["PETR4.SA"],
    "symbols_pending": ["VALE3.SA", "ITUB4.SA"],
    "start_date": "2018-01-01",
    "end_date": df.index[-1].strftime("%Y-%m-%d"),
    "records": {"PETR4.SA": len(df)},
    "source": "yfinance (Yahoo Finance)",
    "note": "PETR4 coletado na Fase 4. VALE3 e ITUB4 serão adicionados posteriormente.",
}

with open(raw_dir / "collection_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Metadata salvo com sucesso!")
print(json.dumps(metadata, indent=2, ensure_ascii=False))
