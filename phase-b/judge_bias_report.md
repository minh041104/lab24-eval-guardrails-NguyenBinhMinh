# Judge Bias Report

## Quantified Bias Table

| Bias | Measurement | Result | Mitigation |
|---|---:|---:|---|
| Position bias | A wins in first-order run | 2/30 = 6.7% | Swap-and-average; disagreements become tie. |
| Position bias | B wins in first-order run | 28/30 = 93.3% | Inspect if either side exceeds 55%. |
| Length bias | Longer answer wins non-tie comparisons | 19/28 = 67.9% | Keep conciseness in rubric and cap answer length. |

## Final Winner Distribution

| Winner | Count |
|---|---:|
| A | 1 |
| B | 27 |
| tie | 2 |

## Calibration

- Cohen's kappa: 1.000
- Interpretation: Almost perfect agreement

## Conclusion

The judge should be used with swap-and-average and a tie option. The current eval set also needs better human labels before the judge is used as a hard production gate.
