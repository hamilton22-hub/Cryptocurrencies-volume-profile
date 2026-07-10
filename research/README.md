# TBX / GF research toolkit

Исследование фильтров отрицательных сделок для breakout/trend системы ETHUSDT.

## Данные

- `data/gf_on.json`, `data/gf_off.json` — экспорты сделок
- `data/klines|funding|metrics` — Binance Vision (скачиваются скриптом, в git не коммитятся)

## Запуск

```bash
pip install -r research/requirements.txt
python research/scripts/download_binance.py
python research/scripts/analyze_filters.py
python research/scripts/validate_oi_holdout.py
```

Отчёт: `reports/negative_trade_filter_research.md`
