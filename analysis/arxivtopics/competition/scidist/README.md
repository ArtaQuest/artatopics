# Science-Distribution: does the sky time what science studies?

How the world's attention divides across **251 research fields**, 1858-2025, and whether each
field's slice of the pie grows.

## The task
For each (field, year) in `test.csv`, predict the probability that the field holds a **larger share
of citations next year than this year**. Trending is relative by construction: a field only gains
slice by outgrowing the field as a whole.

## The rule that makes this a test of astrology
At prediction time you may use **the field's identity and the date, and nothing else**. `shares.csv`
stops at the wall, so the field's recent state in a test year is not available. Every feature you
build is therefore a function of a birth chart and a transiting sky.

## The metric: mAUC
One ROC-AUC per **field**, across that field's own test years, averaged over fields. This is
deliberate. A per-field constant — "this field usually grows" — scores exactly 0.5, because it
cannot order a field's own years. The model has to say *when*.

## The wall
The last 20% of years (1991-2024) are held out. Every training year precedes every
test year. Choose hyper-parameters on the training span only; the wall is not a validation set.

## Files
| file | what |
|---|---|
| `train.csv` | id, field, year, share, target — every year before 1991 |
| `test.csv` | id, field, year — the held-out span, no share and no label |
| `shares.csv` | the full pie, fields x years, train years only |
| `ephemeris.csv` | ecliptic longitude of Mars, Jupiter, Saturn, Uranus, Neptune, Pluto and the lunar node, 1700-2055 |
| `sample_submission.csv` | id, target |

## What is already known
Published baselines on this exact split, all selected on train and scored on the held-out span:
a per-field constant scores **0.5000** by construction; a purely calendar feature (a transiting
position, identical for every field in a year) scores **0.5275**; and the best of 3,683 classical
astrological and numerological features selected by train performance scores **0.5061**, with
corr(train, held) = **+0.024** across the whole catalogue. Beating 0.5275 with a model selected
honestly on train is an open problem.

Data derived from OpenAlex (CC0). Ephemeris computed from standard orbital elements.
No causal claims are made.
