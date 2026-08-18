# Science-Distribution v3: given the year, predict the pie

**Input: the year. Output: that year's distribution of the world's citations over 251 research fields.**

| file | shape | what |
|---|---|---|
| `train.csv` | 134 rows × 252 | `year` + 251 field columns; **each row sums to 1** — the pie, 1858–1991 |
| `test.csv` | 34 rows × 1 | `year` only — the last 20% of years, 1992–2025; predict the whole row |
| `sample_submission.csv` | 34 × 252 | uniform 1/251 |
| `ephemeris.csv` | 1700–2055 | ecliptic longitudes of Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, lunar node |
| `fields.csv` | 251 | column name → full field name |

The year is the only input. A model that reaches the pie through the sky (`ephemeris.csv` is a function
of the year) is an astrological model; one that uses the year directly is a trend model. Both are
allowed — that is the question the benchmark asks. Every training year precedes every test year.

## Baselines on this split (train-selected, scored on the held-out years)
| model | cross-entropy (nats) | rank ρ |
|---|---|---|
| uniform | 5.5255 | — |
| the train-mean pie (climatology) | 5.1324 | 0.810 |
| sky softmax, selected on train | 5.3499 | 0.848 |
| carry-forward (has memory; reference only) | 4.8715 | 0.946 |

Data derived from OpenAlex (CC0). No causal claims are made.
