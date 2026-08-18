# Science-Distribution · Astro Challenge — what the entry found

**Competition:** kaggle.com/competitions/science-distribution-astro (org 5418) · **dataset:** artafather/science-distribution-251 (public) · **kernel:** artafather/science-distribution-astro-entry

Wall 1991 · train 25,896 rows (1858–1990) · test 8,534 rows (1991–2024) · metric mAUC (one AUC per field, averaged).

## Walk-forward stability selection, 806 features, four inner walls (1967/73/79/85)

| walls on which the feature beats the calendar bar | features |
|---|---|
| 0 of 4 | 484 |
| 1 of 4 | 265 |
| 2 of 4 | 54 |
| 3 of 4 | 3 |
| **4 of 4** | **0** |

Best regularised ensemble (ridge logistic, C chosen on the walls) on a sanity wall inside train: **0.4926** vs calendar bar **0.5160**.

## Calendar + λ·sky, λ chosen on the walls

Walls chose **λ = 2.0** (calendar alone 0.5026 → 0.5253 on the walls, +0.0227). On the true held-out span:
calendar alone **0.4879** · calendar + 2·sky **0.4890** · sky alone **0.4937**. All below 0.5.

## Why: the sign flips across the wall

| transiting body | direction fixed on train → held | oracle direction → held | train mAUC |
|---|---|---|---|
| Mars | 0.4879 | 0.5121 | **0.5295** |
| Neptune | 0.4784 | 0.5216 | 0.5149 |
| lunar node | 0.4822 | 0.5178 | 0.5005 |
| Pluto | 0.5216 | 0.5216 | 0.5162 |
| Saturn | 0.5150 | 0.5150 | 0.5063 |

Mars — the strongest calendar feature on train — is the worst on held; Mars, Neptune and the node
are symmetric around 0.5 between the train direction and the oracle direction. The relationship
between the sky and which fields grow **inverted** between 1961–90 and 1991–2024. Nothing selected
honestly on train can survive that, and this is the mechanism behind every "you learn the era you
select in" result in the campaign.

**Status:** the Kaggle competition cannot score until the evaluation metric (AUC) is set in the
host UI — the API has no field for it and `solution create` / `submit` return 500 / 400 until then.
`submission_calendar.csv` and `submission_shrunk.csv` are ready to submit the moment it is.
