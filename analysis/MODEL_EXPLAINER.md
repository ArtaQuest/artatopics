# Does the sky predict what the world searches for?

*A plain-English tour of the prediction model behind the sidereal trend atlas*

> The **live atlas** is published at **[artaquest.org/research](https://artaquest.org/research/)** — the 87 most
> distinctive topics fitted on **weekly** data (stitched to the monthly backbone; see `weekly_fit.py` /
> `research_site.py`). This document describes the same model; the monthly tool (`trends_fit.py`) is the
> development harness that runs over the full keyword pool.

We built a small statistical model that asks one slightly mad question: if you take 22 years
of what the whole world typed into Google, can you explain its rhythm using the positions of
the planets? The answer, surprisingly often, is "a bit" — and the atlas is the result.

Here is exactly how it works. No magic, just least squares.

## What it predicts

For a single keyword — say *sport*, *finance*, or *medicine* — the model takes the worldwide
Google-Trends monthly search interest from January 2004 to now (≈265 data points) and tries to
**reconstruct that wiggly line from the motion of twelve celestial bodies**. Its outputs are:

- a **zodiac sign** — the 30° slice of the wheel that best fits the topic's rhythm
- a **phase angle** (0–360°) — the precise spot on the wheel it locks onto
- **R²** — how much of the 22-year search history the fit explains
- a **distinctiveness score** ("sig") — how sharply *one* angle wins over all the others
- **twelve per-body R²-contributions** — each body's **non-negative** share of the explained
  variance, from a simple decomposition (flatten the body to its mean, recompute R², take the
  drop), normalised to sum **exactly to R²**. (Shown on the pages instead of the raw weights — see
  the verification below for why.)

## The inputs: planets as clocks

The model treats each of twelve bodies as a clock ticking at its own orbital period:

| Body | Period | | Body | Period |
|------|--------|-|------|--------|
| Moon | 27 days | | Jupiter | 11.9 yr |
| Mercury | 88 days | | Saturn | 29.5 yr |
| Venus | 225 days | | Chiron | 50 yr |
| Sun | 1 yr | | Uranus | 84 yr |
| Mars | 1.9 yr | | Neptune | 165 yr |
| Lunar node | 18.6 yr | | Pluto | 248 yr |

A sidereal ephemeris gives each body's longitude (0–360°) for every month since 2004. Fast
bodies sweep round many times; slow ones barely move — so together they span rhythms from
"about a month" to "centuries".

## The method: a phase sweep, fit by least squares

The whole model is **exactly 14 numbers**: twelve body-weights, one baseline (intercept), and
one shared phase angle. That last one is the trick.

We **sweep a candidate angle `φ` around the zodiac** — 0°, 5°, 10°, … 355°, so 72 positions in
all. At each angle we build one feature per body using a *sinc* bump:

```
feature(body, t) = sinc( Δλ_rad(t) / T_body )      where  Δλ_rad = deg2rad( wrap(λ_body(t) − φ) )
```

Here `np.sinc(x) = sin(πx)/(πx)` — the *normalised* sinc, the textbook ideal-interpolation
kernel from the sampling theorem. `Δλ_rad` is the wrapped angular distance from the body to the
trial angle (in radians, range ±π), and `T_body` is the orbital period in years, acting as a
bandwidth. The sinc gives each body a **peak of 1 when it sits exactly at `φ`**, with the first
zero at `Δλ = T_body radians`, i.e. at a half-width of **`rad2deg(T_body)` degrees**.

### What that actually means for each body (verified against the real ephemeris)

The half-width `rad2deg(T)` has a sharp consequence — *most bodies never reach their first zero
inside the ±180° that angles can span*:

| Body | T (yr) | first-zero half-width | character of the feature |
|------|--------|----------------------|--------------------------|
| Moon | 0.075 | **4.3°** | razor-thin spike — but see aliasing below |
| Mercury | 0.24 | 13.8° | narrow bump |
| Venus | 0.62 | 35.2° | moderate bump |
| Sun | 1.0 | 57.3° | broad bump (≈ annual season) |
| Mars | 1.88 | 107.8° | very broad bump |
| Jupiter | 11.9 | 680° | **no zero** — near-constant ramp |
| Saturn…Pluto | 29–248 | 1 700–14 000° | **no zero** — essentially constant |

So "narrow for fast, broad for slow" is true, but the slow half is more extreme than the phrase
suggests: from Jupiter outward the sinc **never completes even a quarter-bump** over the
observable angle range. Those columns are not band-pass features at their orbital period — over
22 years of data we never see a 165-year cycle *cycle*. They are smooth, near-monotonic ramps
that act as a **low-frequency trend basis**, not as rhythms.

Then at each angle we fit

```
search(t) ≈ baseline + Σ_body  weight_body · feature(body, t)
```

by **ordinary least squares** (negative weights allowed, no regularisation). Ordinary least
squares *is* the minimiser of **mean-squared error (MSE)**, so to keep training and tuning on the
**same loss** we also pick the angle by MSE: we compute the MSE at all 72 angles and keep the
**angle of smallest MSE**. Because the data variance is fixed across angles, that is identically
the **angle of largest R²** — so the headline R² is genuinely the best the model achieves, not a
number reported at an angle chosen by some other criterion. That winning angle's 30° slice is the
topic's **sign**:

```
sign = SIGNS[ floor(φ / 30) mod 12 ]      φ = argmin_φ MSE(φ) = argmax_φ R²(φ)
```

## Verification: physics & signal processing (what holds up, what doesn't)

I checked the model's claims numerically against the actual ephemeris and fitted weights. The
machinery is sound; some of the *interpretation* on the pages is not.

**Holds up**

- The sinc kernel and its bandwidth are implemented correctly. First-zero half-widths match
  `rad2deg(T)` exactly (Moon 4.3°, Sun 57.3°, slow bodies > 180°).
- The Sun's best-phase feature correlates **0.62** with a pure 12-month seasonal cosine — i.e.
  the strongest *genuinely varying* body is essentially encoding ordinary calendar seasonality,
  which is a real and expected driver of search interest.

**Does not hold up — three concrete problems**

1. **Aliasing (Nyquist).** The data is monthly, so the fastest resolvable period is 2 months
   (0.17 yr). The Moon (27 d) and Mercury (88 d) cycle *faster than we sample* — their monthly
   columns are aliased noise. Measured: the Moon's feature contributes ~0.3% of the fitted
   wiggle and correlates only 0.13 with anything seasonal. Harmless, but meaningless.

2. **Near-degenerate slow columns → exploding weights.** From Jupiter outward the features are
   near-constant (measured std **≈ 1e-4 to 0** over the window). The design's condition number is
   ~3×10⁶. Least squares responds with **astronomically large cancelling weights** — e.g. for
   *Science*, Pluto = −1.2×10⁷ and Neptune = +7.1×10⁶. These aren't physical "shares"; they're the
   fit sculpting a smooth trend out of two nearly-identical constant columns.

3. **The per-cycle breakdown used to be mislabelled — now fixed.** The pages previously reported
   each cycle's "share of the predicted rhythm" as `|weight| / Σ|weight|`, which is meaningless
   when the slow bodies carry giant cancelling weights. They now use a simple variance
   decomposition: **flatten each body's feature to its mean** (keeping the fitted weights),
   **recompute the empirical R²**, and take the drop, `v_b = R²_full − R²(b→mean) =
   Var(βb·fb)/Var(s)`. Each `v_b` is a variance, so **non-negative**, and the twelve are normalised
   to sum **exactly to R²** — no negative or >100% figures to decode. The honest readout is that a
   pair of near-identical slow ramps usually carries most of it (e.g. *Science*: Neptune 50.0%,
   Chiron 35.9% … = R² 88.1%), confirming the R² is mostly a slow multi-year **trend** plus the
   Sun's annual **seasonality**, not a planetary cycle. (This simple measure doesn't disentangle
   the variance two collinear bodies *share* — a Shapley decomposition would, at the cost of an
   exponential refit; we chose the simpler, fully transparent measure.)

**R² is in-sample (and the code now says so).** `fit_topic` computes R² on the exact months it
fits; `DROP_LAST = 12` removes the most recent year from *both* the inputs and the target before
fitting, so that year is excluded entirely — it is **not** a held-out test set. Given the
over-flexible slow-body basis, an in-sample R² is optimistic, so the headline percentages should
be read as "how well it can be made to fit", not "how well it predicts". (An earlier docstring
wrongly described a train/test split; that has been corrected.)

**Physics verdict.** There is no physical mechanism here, and the model doesn't claim one — good.
But it isn't really fitting *celestial cycles* either: the only bodies whose cycles are both
fast enough to resolve and slow enough not to alias are the Sun, Venus and Mars, and the Sun's
contribution is plain seasonality. The slow bodies serve as an arbitrary smooth-trend basis. The
sinc-with-period-bandwidth choice is a reasonable localised kernel, but mixing *radians* of angle
with *years* of period in one ratio is dimensionally ad-hoc — a von Mises (circular Gaussian)
kernel, or measuring Δλ in cycles, would be cleaner. The honest summary: this is a transparent
**curve-fit of a trend + a season + a phase search**, dressed in celestial coordinates.

## The scores: R² and representativeness

- **R²** — `1 − Σ(actual − predicted)² / Σ(actual − mean)²` — the headline "explains X% of its
  22-year search history" shown on each page.
- **Representativeness** (the atlas ranking metric, in [0, 100]) — amplitude × area-share:

  `rep = R² · ( area of the tuning curve inside the winning 30° sign ÷ total area under it )`

  where the *tuning curve* is `τ(φ) = MSE_max − MSE(φ) ≥ 0` (fit quality at each angle, peaking at
  the winner). **Amplitude** is R² (it also suppresses sparse/noisy keywords, which fit poorly);
  the **area-share** is what fraction of the fit-quality "area under the curve" falls inside the
  assigned sign rather than spread around the zodiac. High only when the fit is strong *and*
  concentrated in one sign. (Because the tuning curves are broad, the area-share is modest, so rep
  ranges ~0–18% rather than reaching 100 — it's a ranking score, not a fit percentage.)

  *Why not "sharpness" / impulse-similarity?* The obvious distinctiveness measure — how impulse-like
  the tuning curve is (peak ÷ spread) — turns out to **reward noise** here: measured across the 411
  topics it correlates **−0.63** with R², and its top scorers are near-empty Trends series (a flat
  line with one accidental spike looks impulse-like). Genuine fits have *broad* tuning because the
  slow-body trend fits at every angle. So sharpness ≠ quality in this model.

## Guarding against fooling ourselves

A model with this much freedom can find patterns in noise, so:

- We **drop the most recent 12 months** before fitting (`DROP_LAST = 12`) — real-time Trends data
  is volatile and recency-biased, and excluding it keeps the fit honest.
- We only fit **single-word keywords** — a multi-word phrase is a noisy, ambiguous Trends query.
- Every page carries the same banner, and so does this post:

> **Correlation is not causation.** This is a statistical curiosity. It only measures whether two
> rhythms happen to line up over 22 years. Nothing here claims the sky *causes* anything, or that
> astrology is real.

## How the atlas fills itself: the discovery loop

The companion script doesn't just fit a fixed list — it **grows a balanced atlas**. It watches
how many topics have landed on each of the twelve signs, targets the **thinnest** signs, and pulls
candidate words from a themed pool (~1000 single nouns, each grouped by astrological flavour:
Aries → courage/war/sport, Virgo → hygiene/nutrition, Pisces → dream/ocean/poetry…). It fits each
word and **keeps only the ones whose winning sign is currently thin** — so the distribution
self-balances toward roughly even coverage. Every fit it ever tries is logged, and the best by
goodness-of-fit feed the public top-100 list.

## So — does the sky predict search?

Sometimes the fit is genuinely tight and distinctive; often it isn't. That gap is the honest,
interesting part. The model is a clean, transparent piece of machinery — twelve clocks, a sweep
of angles, and least squares — and it lets anyone see, topic by topic, exactly how well a cosmic
rhythm happens to line up with a human one. What you do with that coincidence is up to you.

*Source: [analysis/trends_fit.py](trends_fit.py) (the fit) and [analysis/discover.py](discover.py) (the balanced discovery loop)*
