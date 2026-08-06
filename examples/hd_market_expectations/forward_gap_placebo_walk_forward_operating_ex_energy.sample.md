# Forward Gap Lambda Policy Validation

As of: `2024-12-31`
Validation mode: **walk_forward**
Walk-forward splits: train ends `2014-12-31, 2016-12-31, 2018-12-31, 2020-12-31`, test window `2` years
Placebo runs: **3** (shuffle gap values within each driver/horizon panel before scoring the same validation split)
Candidates attempted: **84**
Successful driver/horizon evaluations: **912**
Excluded sectors: **10 Energy, 40 Financials, 55 Utilities, 60 Real Estate**

## Global Lambda

Best lambda: `0.5`
Mean MAE improvement: `0.007413079306394106`
Stable lambda `0.5` mean MAE improvement: `0.007413079306394106`

| Scope | Best lambda | Eval count | Mean MAE improvement | Pass rate |
|---|---:|---:|---:|---:|
| balance_sheet:1Y | 0.8 | 46 | 0.009426001121214672 | 0.5434782608695652 |
| balance_sheet:2Y | 1.0 | 46 | 0.016849984619269544 | 0.5652173913043478 |
| capital_efficiency:1Y | 1.0 | 1 | 0.015327586264884618 | 1.0 |
| capital_efficiency:2Y | 0.8 | 1 | 0.012917128594530314 | 1.0 |
| cash_conversion:1Y | 0.1 | 39 | -0.0005592925220244996 | 0.358974358974359 |
| cash_conversion:2Y | 0.35 | 39 | 0.004466469628785725 | 0.6923076923076923 |
| cash_margin:1Y | 0.5 | 21 | 0.014509210556127316 | 0.7619047619047619 |
| cash_margin:2Y | 0.35 | 21 | 0.011985615366864176 | 0.6666666666666666 |
| growth:1Y | 0.35 | 185 | 0.005477723823706469 | 0.6756756756756757 |
| growth:2Y | 0.35 | 185 | 0.0023758326106086327 | 0.5135135135135135 |
| margin:1Y | 0.65 | 164 | 0.014274419544686326 | 0.6951219512195121 |
| margin:2Y | 0.65 | 164 | 0.012161836404387892 | 0.6524390243902439 |

## Placebo Check

| Lambda | Actual mean MAE improvement | Placebo mean MAE improvement | Actual - placebo | Actual beats placebo |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.1 | 0.0026280427089779403 | -2.4338462848346038e-05 | 0.002652381171826286 | 0.6907894736842105 |
| 0.2 | 0.0047034534369894705 | -8.421078385617326e-05 | 0.004787664220845644 | 0.6820175438596491 |
| 0.35 | 0.006736486452875759 | -0.00023917047566385715 | 0.006975656928539615 | 0.6557017543859649 |
| 0.5 | 0.007413079306394106 | -0.00047210614981628993 | 0.007885185456210397 | 0.6348684210526315 |
| 0.65 | 0.00683811461493029 | -0.0007822692433687763 | 0.007620383858299066 | 0.6118421052631579 |
| 0.8 | 0.005002640862583673 | -0.0011702856876767617 | 0.006172926550260435 | 0.5756578947368421 |
| 1.0 | 0.0006732746688570549 | -0.0018088616809599973 | 0.0024821363498170527 | 0.5208333333333334 |

## Sector Check

| Sector | Best lambda | Eval count | Mean MAE improvement | Pass rate |
|---|---:|---:|---:|---:|
| 15 Materials | 0.8 | 138 | 0.019597420673575193 | 0.6666666666666666 |
| 20 Industrials | 0.2 | 152 | 0.0014632659279525855 | 0.5263157894736842 |
| 25 Consumer Discretionary | 0.65 | 114 | 0.014271011436164273 | 0.7719298245614035 |
| 30 Consumer Staples | 0.2 | 148 | 0.000786396357148593 | 0.6148648648648649 |
| 35 Health Care | 0.5 | 116 | 0.010536303313209691 | 0.6896551724137931 |
| 45 Information Technology | 0.65 | 116 | 0.010708313096124912 | 0.6120689655172413 |
| 50 Communication Services | 0.5 | 128 | 0.004482100844784357 | 0.5703125 |

## Recommended Policy

```json
{
  "by_family_horizon": {
    "cash_conversion:1Y": 0.1,
    "cash_margin:2Y": 0.35
  },
  "default_lambda": 0.5,
  "fallback_lambda": 0.5,
  "min_group_evaluations": 20,
  "min_mean_lift_vs_default": 0.001,
  "min_pass_rate_lift_vs_default": 0.15
}
```
