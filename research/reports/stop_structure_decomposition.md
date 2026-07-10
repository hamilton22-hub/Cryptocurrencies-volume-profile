# Stop-structure decomposition — deprecated

Этот отчёт заменён на
[`causal_policy_research.md`](./causal_policy_research.md).

Причины:

- старый path для большинства 1H начинался от signal-time, а не fill-time;
- `stop_hunt` был circular label: будущий `CF MFE≥1R` входил в определение
  класса;
- будущий максимум MFE ошибочно использовался как реализуемый exit;
- emergency barrier и intrabar ordering не моделировались;
- восстановленный `limit_retest` не являлся реальным типом ордера.

Числа `95 stop_hunt`, `+225R @ cap2` и разрезы `1H limit_retest` нельзя
использовать для изменения риска или production-логики.

Новый анализ принимает решения только до касания исходного стопа, использует
first-passage policy и оценивает paired `ΔR` относительно canonical результата.
