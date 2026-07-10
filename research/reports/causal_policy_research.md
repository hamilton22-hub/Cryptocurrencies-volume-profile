# Причинное исследование стопов и ранних выходов

## Короткий итог

После устранения look-ahead и сверки единиц R:

1. **Расширять stop сейчас нельзя.** На 15M осталось всего 10 сопоставимых
   stop-activation событий для основного checkpoint. Always-extend дал лишь
   `+0.44R`; предиктивную модель на таком N оценить нельзя.
2. Предвходовый hopeless-фильтр снова ухудшил систему:
   `−13.39R` на 15M и `−54.97R` на 1H proxy.
3. Через 2–8 закрытых 15M-баров будущий проигрыш уже предсказывается
   (`AUC≈0.67→0.81`), но ML-выходы всё ещё задевают хвост.
4. Лучший исполнимый shadow-кандидат — **15M failed expansion после двух
   полных post-fill баров**. Он дал `+5.33R`, но family-adjusted
   `p≈0.76`: статистически не доказан.
5. Giveback после +1R дал небольшие положительные gross-результаты на
   reconciled subset, но нестабилен, задевает tail и не проходит поправку
   (`family p≈0.53`).

Production-изменений не рекомендуется. Ниже — конкретные предложения для
shadow по каждому направлению.

---

## 1. Почему прежние stop-hunt числа сняты

Старые `95 stop_hunt`, `AUC≈0.80` и `+225R @ cap2` были неисполняемым oracle:

- у большинства 1H path начинался от signal-time, а не от фактического fill;
- будущий `CF MFE≥1R` входил в определение класса;
- будущий максимум MFE использовался как будто это реализуемый exit;
- признаки `beyond_stop`/`wick_only` знали будущий путь;
- emergency barrier не участвовал в competing race;
- target мог ошибочно засчитываться до активации extension.

Старый отчёт помечен deprecated.

---

## 2. Что теперь считается причинным

- решение — только после полностью закрытой 15M-свечи;
- decision-time строго раньше canonical exit;
- исходный stop не должен быть проторгован до decision-time;
- target не засчитывается до stop-activation;
- activation-bar считается консервативно;
- partial reduction исполняется по следующему open, не по уже известному close;
- train включает только outcomes, завершившиеся до cutoff;
- threshold и модель остаются в одной calibration scale;
- same-bar target/emergency: adverse-first;
- gap исполняется по наблюдаемому open, поэтому `−1.25R` — номинальный barrier,
  а не гарантированный worst loss.

### Главный data gate: Signal R и price R не совпадают

Экспортированный `r` нельзя безусловно вычитать из синтетического price-R:

- 15M: только 295 из 709 сделок совпадают в пределах `0.10R`;
- 1H: 331 из 469, но fill-time всё равно proxy;
- крупные tails часто содержат pyramiding/partials и особенно расходятся.

Поэтому policy economics считается только для строк, где:

```text
abs(exported_r − side × (exit_price − entry_price) / stop_distance) <= 0.10R
```

Остальные строки остаются в диагностике, но не получают synthetic `ΔR`.

### Clock

- 15M: 707 сделок имеют entry-price в timestamp-баре;
- 1H: 469 first-touch proxy, медианный лаг 2×15m;
- checkpoint `2/4/8 bars` означает столько полных баров после fill-бара;
- фактическое время решения находится примерно в диапазонах
  `30–45 / 60–75 / 120–135 минут` после fill.

1H path-результаты не production-grade.

---

## 3. Направление A: stop-hunt / extension

Проверены checkpoints `−0.65/−0.75/−0.85R`, emergency `−1.25/−1.50R`,
target `0/+0.5/+1R`, horizon 12h.

Policy меняет canonical logic только после original-stop activation. Если
stop не активировался, full-position `ΔR=0`.

### 15M, emergency −1.25R, target +0.5R

| Checkpoint | Decisions | Economic rows | Reconciled stop activations | ΔR |
|---|---:|---:|---:|---:|
| −0.65R | 50 | 29 | 10 | **+0.44R** |
| −0.75R | 27 | 18 | 8 | **−0.72R** |
| −0.85R | 5 | 3 | 1 | **−0.28R** |

Это не доказательство пользы q=−0.65: всего два beneficial outcomes, и
train/test недостаточны даже для честного AUC.

### `LOW_ENERGY_ADVERSE_PROBE` перепроверен

Гипотеза:

```text
15M, close <= −0.65R
volume_since_fill <= discovery median
directional taker flow не сильнее −0.043 против позиции
```

После исправлений:

- 14 decisions;
- 9 stop proxies, но только 4 reconciled;
- discovery 2021–2022: **−0.62R**;
- OOS 2023–2026: `+2.90R`;
- full sample: `+2.28R`;
- unadjusted monthly block `p≈0.25`;
- ни одно правило этого семейства не имело положительного discovery `ΔR`.

**Вердикт:** даже shadow-правило не замораживать. Можно логировать как
mechanistic feature, но не считать найденным эджем.

### Предложение по stop-hunt

Пока не менять execution. Нужен event log:

- точный fill timestamp/price;
- фактический stop trigger timestamp и trigger type;
- size и partials;
- mark/last price;
- stop amend ACK/latency;
- fees, funding, slippage;
- 1m или tick path вокруг stop.

После этого тестировать заранее фиксированную state machine:

```text
checkpoint −0.65R
  -> KEEP_BASE по умолчанию
  -> EXTEND только при frozen high-confidence score
  -> nominal emergency −1.25R
  -> не более одного extension
  -> atomic amend; reject/timeout => KEEP_BASE
  -> отдельный risk budget для gap-through
```

---

## 4. Направление B: hopeless до входа

Оптимизировалось действие `skip`, а не accuracy:

```text
ΔR_skip = 0 − canonical R
```

Expanding-year OOS:

| Контур | Actions | ΔR | Затронутые >5R |
|---|---:|---:|---:|
| 15M pre-entry | 31 | **−13.39R** | 1 |
| 1H signal/fill proxy | 32 | **−54.97R** | 5 |

**Вердикт:** предвходовый hopeless-фильтр отклонить.

---

## 5. Направление C: hopeless/ambiguous по early path

### Классифицировать будущий loss уже можно

15M OOS:

| Полных post-fill баров | Eventual negative AUC | Toxic ≤−0.5R AUC |
|---:|---:|---:|
| 2 | 0.67 | 0.63 |
| 4 | 0.72 | 0.69 |
| 8 | **0.81** | **0.81** |

Но outcome prediction и полезное действие — разные задачи.

### ML action-value replay

| Checkpoint | Actions | ΔR | Затронутые >5R | Active positive folds |
|---|---:|---:|---:|---:|
| 2 bars | 42 | +8.23R | 2 | 2/3 |
| 4 bars | 71 | +10.31R | 2 | 2/3 |
| 8 bars | 62 | +7.47R | 2 | 2/3 |

Положительный aggregate недостаточен:

- 2-bar folds: `+7.51, +6.42, −5.70R`;
- 4-bar folds: `+4.07, +6.28, −0.03R`;
- в 2026 модель не открывала действий;
- каждый контур всё ещё убил два >5R winner;
- этот model/threshold search не имеет untouched confirmatory holdout.

Для требования «не затронуть положительные» ML пока не проходит.

### Лучшее простое shadow-правило: `FAILED_EXPANSION_2B`

```text
TF = 15M
через 2 полных post-fill бара:
  current R <= +0.25
  MFE <= +0.50R
  cumulative directional taker imbalance против позиции
exit = следующий 15M open
```

OOS 2023–2026:

- 58 actions;
- `+5.33R`;
- ни одного >5R tail;
- максимум затронутого winner: +2.48R;
- по годам: `+1.04, −1.51, +2.49, +3.32R`;
- unadjusted block `p≈0.07`;
- family-adjusted по 64 rules `p≈0.76`.

**Вердикт:** лучший кандидат для frozen shadow, но не для production.

### Предлагаемая ambiguous state machine

```text
high-confidence failed expansion -> shadow EXIT_NEXT_OPEN
high-confidence rescue           -> пока не найден
middle / ambiguous               -> ABSTAIN, canonical logic
```

Default всегда `ABSTAIN`. Нельзя заставлять каждую сделку попасть в
hopeless/rescue.

---

## 6. Направление D: structural early exits / giveback

Проверено:

```text
arm после +1R
после более позднего close <= 0 / +0.25 / +0.5R
exit на следующем open
```

Discovery выбрал threshold `+0.5R`:

- discovery: `+4.88R`;
- OOS: `+2.32R`;
- full sample: `+7.20R` на 56 actions;
- задет один >5R winner;
- годы: `+3.70, +1.18, −2.88, +4.94, +2.32, −2.07R`;
- unadjusted `p≈0.43`, family-adjusted `p≈0.53`.

Более консервативный `close<=0R` дал +2.06R на 15 событиях и не задел >5R,
но это full-sample exploratory результат, не выбранный подтверждающим
процессом.

**Вердикт:** production giveback отклонить. Логировать `ARM1_CLOSE0` как
вторичный shadow-контур можно, но не смешивать его статистику с
`FAILED_EXPANSION_2B`.

---

## 7. Что именно предлагается делать

### Shadow 1 — основной

`FAILED_EXPANSION_2B`, только 15M, формула выше.

### Shadow 2 — диагностический

`ARM1_CLOSE0`, только 15M:

```text
arm после +1R
если более поздний completed close <= 0R
shadow exit next open
```

### Не включать

- stop widening;
- pre-entry hopeless filter;
- 1H path filter;
- ML early exit;
- giveback +0.25/+0.5R.

### Логировать для каждого shadow action

- exact fill/decision/canonical-exit timestamps;
- exported R и price-R;
- size/partials/pyramids;
- current R, MFE, MAE;
- volume/taker/OI snapshot и source timestamp;
- hypothetical next-open fill;
- fees/slippage/funding;
- canonical и shadow result.

---

## 8. Acceptance gates

Для каждого shadow-контура отдельно:

- формула и thresholds не меняются до конца forward OOS;
- минимум 50 actions;
- paired `ΔR_net > 0` в обеих половинах;
- 95% monthly-block CI > 0;
- положительный результат при execution penalty 0.10R;
- `tail >5R` loss = 0;
- max forgone winner заранее ограничен;
- MDD/ES считаются по event-time portfolio replay, не entry-order cumsum;
- после shadow используется только новый forward OOS.

Для stop-extension дополнительно требуется минимум 30 reconciled
original-stop activations и order-level replay.

---

## 9. Ограничения данных

- 1H fill-time восстановлен эвристически;
- нет size/partial/pyramid ledger;
- в kline-файле 11 gaps >15m;
- в aggregate OI-файле 1812 gaps >5m;
- 15M OHLC не восстанавливает intrabar ordering;
- история до 2026 уже многократно исследована и больше не является untouched
  holdout.

Хэши входов, версии библиотек, coverage и random seed записаны в
`artifacts/causal_run_manifest.json`.

---

## Окончательный вердикт

| Направление | Решение |
|---|---|
| Stop-hunt extension | данных недостаточно, не менять |
| Hopeless до входа | отклонить |
| 15M early failed expansion | frozen shadow |
| Ambiguous | abstain / canonical |
| Structural giveback | только диагностический shadow |
| 1H execution | ждать order log |

Главный вывод: будущий проигрыш действительно становится видимым по раннему
path, но доказать полезность конкретного exit намного сложнее. Primary target
должен оставаться `paired net ΔR` с защитой хвоста, а не AUC.
