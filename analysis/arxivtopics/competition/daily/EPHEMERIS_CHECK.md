# Triple-check of the ephemerides and the phasor math (2026-08-18)

Three engines compared at 2000-07-01 12:00 UT: **kerykeion** (Swiss Ephemeris), **PyJHora**, and the
campaign's own `arxiv_fit.sky_lunar` table.

## Findings
1. **The campaign's ephemeris is correct — and it is SIDEREAL (Lahiri).** It matches kerykeion sidereal
   to 0.1–0.7° across 1800–2050 (Mars loosest at −0.66°, Uranus/Neptune/Pluto ±0.03°) and sits exactly one
   ayanamsa (20.4° in 1800 → 24.5° in 2050, growing at precession) off kerykeion tropical. Every campaign
   result was therefore computed in the sidereal frame, consistently. The node matches the true node ±1°.
   Prose that called it "tropical" was wrong; the numbers were not.
2. **PyJHora's `sidereal_longitude` index order is NOT Sun,Moon,Mars,Mercury,Jupiter,Venus,Saturn,Rahu,Ketu.**
   Matched against kerykeion, indices 0–8 are Sun, Moon, Mercury, Venus, **Sun (again)**, Jupiter, Saturn,
   Uranus, Neptune. Mars and the nodes are not in 0–8. Sun/Moon/Saturn agree with kerykeion to 0.006°;
   the labels I had put on indices 2–5,7,8 were wrong. **All PyJHora-based daily results were re-run on
   kerykeion**, one engine, bodies named. (A constant ~0.88° PyJHora–kerykeion offset remains on planets
   other than Sun/Moon/Saturn — true vs apparent position or a Lahiri variant; sub-degree, noted.)
   The yearly `build_charts.py` Vedic table carries the same mislabelling and is superseded.
3. **Ayanamsa:** PyJHora 23.864°, kerykeion 23.860° at 2000-07-01; my earlier constant 23.85° was fine.
4. **The phasor math**, verified numerically: the expansion of |b + Σaᵢe^{i(θᵢ−pᵢ)}|² into
   b²+Σaᵢ² + 2bΣaᵢcos(θᵢ−pᵢ) + 2Σᵢ<ₖaᵢaₖcos((θᵢ−pᵢ)−(θₖ−pₖ)) is exact (1e-15); the linear design recovers
   b, aᵢ, pᵢ exactly from noiseless data; the quadratic b⁴−C₀b²+ΣMᵢ²/4=0 has two roots and only the '+'
   root reproduces the curve. Signs: cos(θ−natal) peaks at the return, sin<0 = applying. **One stated
   approximation:** the campaign's *record* model fits √y ≈ b + Σaᵢcos(θᵢ−pᵢ), which drops the (Σaᵢsin)²
   term of |z|² — second-order in a/b, valid in the shares regime; the daily phasor uses the exact |z|² form.
5. Rates from a 10-day baseline are instantaneous (Mercury retrograde, Jupiter/Saturn near opposition) and
   are not comparable to textbook means — a probe artefact, not physics.

## Daily results, re-run on kerykeion (127 categories, per-category AUC averaged)
| model | AUC |
|---|---|
| ridge logistic on the sky | 0.5365 (was 0.5385 on the mislabelled table) |
| calendar only | 0.5477 |
| memory | 0.5590 |
| circular-shift null | 0.5191 |
| phasor, level / rise, all 8 | 0.4981 / 0.4998 |
| phasor, Sun only, rise | 0.5138 |
Conclusion unchanged: the sky's excess over the shift null is the season, read through the Sun.
