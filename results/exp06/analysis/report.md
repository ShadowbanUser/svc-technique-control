# Experiment 06 analysis

Distances in dB (RMS per mel band of the cepstrally smoothed log-mel difference, level removed). Brackets: 95% cluster-bootstrap interval over the five held-out singers.

## Mean distance to the real technique recording

| Technique | n | Plain recording | Base | Adapter α=0.5 | Adapter α=1 | Matched EQ | Oracle EQ | Static bound | Guidance* | Technique in source |
|---|---|---|---|---|---|---|---|---|---|---|
| Breathy | 50 | 6.82 | 7.96 | 7.23 | 7.22 | 7.51 | 6.31 | 5.87 | 8.54 | 7.22 |
| Belt | 41 | 8.78 | 10.50 | 9.48 | 9.32 | 9.50 | 7.82 | 7.11 | 11.02 | 8.50 |
| Vocal fry | 46 | 11.38 | 12.30 | 11.87 | 10.83 | 11.99 | 10.55 | 9.40 | 14.28 | 9.57 |
| Forte | 48 | 6.02 | 8.06 | 7.37 | 7.37 | 7.62 | 6.36 | 5.96 | 8.29 | 6.84 |
| Piano | 50 | 5.84 | 7.52 | 6.94 | 6.87 | 7.06 | 6.01 | 5.92 | 8.02 | 6.57 |

*Guidance: scales on 'a' only (one exercise per singer).

## Static and dynamic parts (D² = D_static² + D_dyn²)

| Technique | Base static | Base dyn | Adapter static | Adapter dyn | Adapter − base, dyn | Correct − wrong labels, dyn | Static share of drop in D² |
|---|---|---|---|---|---|---|---|
| Breathy | 5.31 | 5.87 | 4.88 | 5.27 | -0.60 [-1.06, -0.25] | -0.86 [-1.16, -0.56] | +0.43 [-9.98, +4.12] |
| Belt | 7.52 | 7.11 | 6.22 | 6.88 | -0.23 [-0.40, -0.05] | -0.24 [-0.78, +0.44] | +0.87 [+0.60, +1.39] |
| Vocal fry | 7.88 | 9.40 | 7.34 | 7.87 | -1.53 [-1.97, -1.06] | -1.07 [-1.44, -0.59] | +0.24 [-0.20, +0.42] |
| Forte | 5.28 | 5.96 | 4.55 | 5.72 | -0.24 [-0.66, +0.23] | -0.49 [-0.70, -0.27] | +0.73 [+0.41, +1.55] |
| Piano | 4.58 | 5.92 | 4.14 | 5.43 | -0.50 [-0.82, -0.22] | -0.74 [-1.09, -0.34] | +0.39 [-0.03, +0.46] |

## Paired comparisons of the adapter (α = 1), raw distance

| Technique | − base | − matched EQ | − oracle EQ | − static bound | Correct − wrong labels | Gap closed toward source-carried |
|---|---|---|---|---|---|---|
| Breathy | -0.74 [-2.01, +0.21] | -0.29 [-0.57, -0.02] | +0.91 [+0.07, +1.58] | +1.35 [+0.60, +1.88] | -1.73 [-2.46, -1.18] | +1.00 [-0.30, +2.40] |
| Belt | -1.17 [-2.65, +0.33] | -0.17 [-0.25, -0.09] | +1.50 [+0.82, +2.34] | +2.22 [+1.45, +3.39] | -1.42 [-3.17, +0.44] | +0.59 [-0.17, +1.51] |
| Vocal fry | -1.48 [-2.28, -0.64] | -1.17 [-1.67, -0.65] | +0.27 [-0.53, +1.03] | +1.43 [+0.67, +2.16] | -1.70 [-2.45, -1.13] | +0.54 [+0.34, +0.74] |
| Forte | -0.68 [-1.18, -0.04] | -0.25 [-0.65, +0.17] | +1.01 [+0.28, +1.75] | +1.42 [+0.59, +2.23] | -1.68 [-2.00, -1.39] | +0.56 [+0.04, +1.11] |
| Piano | -0.65 [-1.18, -0.17] | -0.19 [-0.37, -0.04] | +0.86 [+0.47, +1.29] | +0.95 [+0.36, +1.61] | -1.97 [-2.73, -1.33] | +0.68 [+0.19, +1.15] |

## Which label brings the output closest to each real technique (counts over items)

**d_raw**

| Real technique \ label | Breathy | Belt | Vocal fry | Forte | Piano | n |
|---|---|---|---|---|---|---|
| Breathy | 36 | 0 | 0 | 0 | 14 | 50 |
| Belt | 0 | 24 | 2 | 10 | 5 | 41 |
| Vocal fry | 4 | 0 | 31 | 1 | 10 | 46 |
| Forte | 0 | 9 | 0 | 25 | 14 | 48 |
| Piano | 5 | 0 | 1 | 5 | 39 | 50 |

**d_dyn**

| Real technique \ label | Breathy | Belt | Vocal fry | Forte | Piano | n |
|---|---|---|---|---|---|---|
| Breathy | 41 | 1 | 1 | 0 | 7 | 50 |
| Belt | 9 | 24 | 8 | 0 | 0 | 41 |
| Vocal fry | 11 | 0 | 35 | 0 | 0 | 46 |
| Forte | 2 | 12 | 0 | 14 | 20 | 48 |
| Piano | 11 | 4 | 0 | 5 | 30 | 50 |

## Pre-specified checks

- O1 Breathy: adapter − base D_dyn -0.60 [-1.06, -0.25] (below zero); correct − wrong -0.86 [-1.16, -0.56] (below zero).
  dyn classifier P(Breathy) − base +0.39 [+0.28, +0.49] (above zero), hit rate 0.54.
- O1 Vocal fry: adapter − base D_dyn -1.53 [-1.97, -1.06] (below zero); correct − wrong -1.07 [-1.44, -0.59] (below zero).
  dyn classifier P(Vocal fry) − base +0.55 [+0.26, +0.75] (above zero), hit rate 0.82.
- O2 Belt: static share of the drop in D² +0.87 [+0.60, +1.39] (mean drop in D² +20.39).
- O2 Forte: static share of the drop in D² +0.73 [+0.41, +1.55] (mean drop in D² +11.35).
- O2 Piano: static share of the drop in D² +0.39 [-0.03, +0.46] (mean drop in D² +9.33).

## Classifier

| Feature set | LOSO bal. acc. | Held-out bal. acc. | Plain + oracle EQ classified as the technique (breathy / belt / fry / forte / piano) |
|---|---|---|---|
| full | 0.62 | 0.67 | 0.16 / 0.56 / 0.00 / 0.02 / 0.06 |
| dyn | 0.60 | 0.63 | 0.16 / 0.26 / 0.00 / 0.04 / 0.02 |

**full: probability of the intended technique minus that of the base output (held-out singers)**

| Technique | Adapter α=0.5 | Adapter α=1 | Matched EQ | Oracle EQ | Guidance | Technique in source |
|---|---|---|---|---|---|---|
| Breathy | +0.07 [+0.05, +0.11] (hit 0.02) | +0.25 [+0.20, +0.31] (hit 0.62) | +0.04 [+0.03, +0.05] (hit 0.00) | +0.06 [+0.03, +0.08] (hit 0.02) | +0.13 [+0.09, +0.18] (hit 0.40) | -0.00 [-0.01, +0.01] (hit 0.00) |
| Belt | +0.24 [+0.18, +0.30] (hit 0.84) | +0.34 [+0.25, +0.42] (hit 0.96) | +0.27 [+0.22, +0.31] (hit 0.92) | +0.16 [+0.08, +0.24] (hit 0.85) | +0.38 [+0.26, +0.49] (hit 1.00) | +0.05 [-0.01, +0.12] (hit 0.44) |
| Vocal fry | +0.05 [+0.03, +0.07] (hit 0.10) | +0.38 [+0.24, +0.50] (hit 0.76) | +0.02 [+0.00, +0.04] (hit 0.02) | +0.01 [-0.01, +0.03] (hit 0.09) | +0.04 [+0.03, +0.06] (hit 0.00) | +0.34 [+0.25, +0.41] (hit 0.61) |
| Forte | +0.05 [+0.03, +0.07] (hit 0.10) | +0.35 [+0.23, +0.44] (hit 0.84) | +0.01 [-0.03, +0.04] (hit 0.02) | +0.01 [-0.02, +0.04] (hit 0.10) | +0.03 [-0.00, +0.06] (hit 0.00) | +0.39 [+0.16, +0.63] (hit 0.58) |
| Piano | +0.03 [+0.02, +0.04] (hit 0.04) | +0.19 [+0.14, +0.24] (hit 0.44) | +0.02 [-0.00, +0.04] (hit 0.00) | +0.06 [+0.03, +0.09] (hit 0.04) | +0.18 [+0.10, +0.27] (hit 0.60) | +0.09 [+0.03, +0.16] (hit 0.08) |

Base outputs classified as: {'straight': 34, 'breathy': 0, 'belt': 10, 'vocal_fry': 3, 'forte': 3, 'piano': 0}

**dyn: probability of the intended technique minus that of the base output (held-out singers)**

| Technique | Adapter α=0.5 | Adapter α=1 | Matched EQ | Oracle EQ | Guidance | Technique in source |
|---|---|---|---|---|---|---|
| Breathy | +0.08 [+0.04, +0.13] (hit 0.04) | +0.39 [+0.28, +0.49] (hit 0.54) | +0.02 [+0.01, +0.04] (hit 0.00) | +0.05 [+0.03, +0.06] (hit 0.00) | +0.09 [+0.04, +0.17] (hit 0.00) | +0.00 [-0.01, +0.02] (hit 0.00) |
| Belt | +0.32 [+0.21, +0.41] (hit 0.80) | +0.37 [+0.25, +0.48] (hit 0.88) | +0.29 [+0.24, +0.34] (hit 0.74) | +0.20 [+0.10, +0.31] (hit 0.78) | +0.31 [+0.12, +0.44] (hit 1.00) | +0.09 [-0.02, +0.23] (hit 0.49) |
| Vocal fry | +0.07 [+0.01, +0.13] (hit 0.14) | +0.55 [+0.26, +0.75] (hit 0.82) | +0.03 [-0.02, +0.06] (hit 0.08) | -0.01 [-0.05, +0.03] (hit 0.07) | +0.03 [-0.01, +0.07] (hit 0.00) | +0.52 [+0.37, +0.67] (hit 0.78) |
| Forte | +0.07 [+0.02, +0.12] (hit 0.14) | +0.49 [+0.33, +0.65] (hit 0.78) | +0.01 [-0.02, +0.04] (hit 0.02) | +0.00 [-0.01, +0.01] (hit 0.06) | +0.08 [+0.02, +0.14] (hit 0.00) | +0.46 [+0.20, +0.74] (hit 0.65) |
| Piano | +0.03 [+0.01, +0.04] (hit 0.06) | +0.24 [+0.16, +0.33] (hit 0.42) | +0.02 [-0.01, +0.05] (hit 0.00) | +0.04 [+0.00, +0.09] (hit 0.02) | +0.19 [+0.08, +0.37] (hit 0.40) | +0.10 [+0.03, +0.17] (hit 0.12) |

Base outputs classified as: {'straight': 28, 'breathy': 0, 'belt': 14, 'vocal_fry': 4, 'forte': 3, 'piano': 1}

## Adapter minus its matched equaliser (Table 7, column 2 is D_dyn)

| Technique | D | D_static | D_dyn |
|---|---|---|---|
| Breathy | -0.29 [-0.57, -0.02] | -0.01 [-0.21, +0.27] | -0.37 [-0.59, -0.17] |
| Belt | -0.17 [-0.25, -0.09] | -0.10 [-0.19, -0.06] | -0.14 [-0.23, -0.01] |
| Vocal fry | -1.17 [-1.67, -0.65] | -0.47 [-0.87, -0.15] | -1.15 [-1.53, -0.69] |
| Forte | -0.25 [-0.65, +0.17] | -0.10 [-0.33, +0.11] | -0.23 [-0.55, +0.11] |
| Piano | -0.19 [-0.37, -0.04] | -0.05 [-0.15, +0.07] | -0.19 [-0.36, -0.07] |
