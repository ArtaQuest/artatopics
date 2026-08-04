# Does the sky predict what the world searches for?

A plain-English (but honest) tour of the prediction model behind our sidereal trend atlas. No magic — just least squares, with every claim checked.

## The question

Take 22 years of what the whole world typed into Google for one keyword. Can you reconstruct that wiggly line of search interest from the positions of the planets? For the **87 most distinctive topics** the model finds the angle of the zodiac whose celestial rhythms best fit 22 years of **weekly** history — browse them all in the live atlas at **[artaquest.org/research](https://artaquest.org/research/)**.

**The data.** Google Trends only returns weekly numbers for windows shorter than ~5 years, each on its own scale, so we fetch overlapping ~4-year chunks and stitch them — rescaling every chunk to the all-time monthly export, then **anchoring each month** so the weekly series matches that monthly backbone exactly while keeping its within-month shape. Topics whose chunks won't scale consistently (sparse, noisy keywords) are dropped.

## The inputs: twelve clocks

A sidereal ephemeris gives each body's longitude $\lambda_b(t)$ (0–360°) for every month since 2004. Each body is a clock ticking at its own orbital period $T_b$:

```
Moon 27d   Mercury 88d   Venus 225d   Sun 1.0y   Mars 1.9y   Node 18.6y
Jupiter 11.9y   Saturn 29.5y   Chiron 50y   Uranus 84y   Neptune 165y   Pluto 248y
```

## The model: a phase sweep over sinc bumps

The whole model is **exactly 14 numbers**: twelve body weights $\beta_b$, one intercept $\beta_0$, and one shared phase angle $\phi$. We sweep $\phi$ over 72 positions ($0°,5°,\dots,355°$). At each angle every body gets one feature — a *normalised sinc* bump centred where the body sits:

$$f_b(t)=\operatorname{sinc}\!\left(\frac{\Delta\lambda_b(t)}{T_b}\right),\qquad \operatorname{sinc}(x)=\frac{\sin(\pi x)}{\pi x}$$

where $\Delta\lambda_b(t)=\lambda_b(t)-\phi$ wrapped into $[-\pi,\pi]$ radians. The feature peaks at 1 when the body is exactly at $\phi$; its first zero is at $\Delta\lambda=T_b$ radians, i.e. a half-width of $\tfrac{180}{\pi}T_b$ degrees — **narrow for fast bodies, broad for slow ones**. Then we fit

$$\hat s(t)=\beta_0+\sum_{b=1}^{12}\beta_b\,f_b(t)$$

## One loss everywhere: MSE

Ordinary least squares **is** the minimiser of mean-squared error. To keep training and tuning on the *same* loss, we also choose the angle by MSE:

$$\operatorname{MSE}(\phi)=\frac{1}{n}\sum_t\big(s(t)-\hat s(t)\big)^2,\qquad \phi^\star=\arg\min_\phi \operatorname{MSE}(\phi)$$

Because the data variance is fixed across angles, $\arg\min_\phi \operatorname{MSE}=\arg\max_\phi R^2$ — so the winning angle is genuinely the best-$R^2$ angle, and the headline score isn't reported at an angle chosen by some other rule:

$$R^2=1-\frac{\sum_t\big(s(t)-\hat s(t)\big)^2}{\sum_t\big(s(t)-\bar s\big)^2}$$

The winning angle's 30° slice is the topic's **sign**: $\ \text{sign}=\big\lfloor \phi^\star/30^\circ\big\rfloor \bmod 12$.

To stay honest we **drop the most recent year** before fitting (real-time Trends data is volatile) and only fit **single-word** keywords. The reported $R^2$ is **in-sample**.

## Crediting the bodies: a simple variance decomposition

A body's raw weight is *not* its importance — the slow bodies barely move over 22 years, so their features are nearly constant and least squares hands them enormous **cancelling** weights (condition number $\approx 3\times10^6$). So instead of weights we use a simple, transparent measure: **flatten each body's feature to its mean** (keeping the fitted weights) and **recompute the empirical $R^2$**. The drop is that body's contribution,

$$v_b=R^2_{\text{full}}-R^2\big(\text{body }b\to\text{mean}\big)=\frac{\operatorname{Var}(\beta_b f_b)}{\operatorname{Var}(s)}\ \ge\ 0$$

Each $v_b$ is a variance, so it can't be negative; we normalise the twelve to sum **exactly to $R^2$**. For example, *Tourism* (Sagittarius, $R^2=97.4\%$):

```
Chiron  49.5     Uranus  1.1
Neptune 28.9     Node    1.1
Saturn  16.2     others ~0.0
                 ----------------
                 Together = 97.4% (= R²)
```

A few slow ramps (Chiron, Neptune, Saturn) carry almost all of it — the honest readout of *where the explained variance sits*, even though this simple measure doesn't disentangle the part those collinear bodies share.

## Ranking topics: representativeness

Which topics belong to their sign most convincingly? We rank by one bounded score that blends **how strong** the fit is with **how concentrated** it is on a single sign. At every trial angle the fit leaves an error; define the **tuning curve** $\tau(\phi)=\operatorname{MSE}_{\max}-\operatorname{MSE}(\phi)\ge 0$ — fit quality versus angle, peaking at the winning angle. Then

$$\text{rep}=R^2\cdot\frac{\sum_{\phi\,\in\,\text{sign}}\tau(\phi)}{\sum_{\phi}\tau(\phi)}$$

amplitude ($R^2$) times the share of the fit-quality **area under the curve** that falls inside the assigned 30° sign rather than spread around the zodiac. It is high only when the fit is both strong *and* concentrated in one sign — and the $R^2$ factor drops sparse, noisy keywords. (We deliberately avoid ranking by tuning-curve *sharpness* / impulse-similarity: in this model a razor-sharp peak signals a near-empty noisy series, not a good fit — genuine fits have broad tuning because the slow-body trend fits at many angles.)

## Recency bias: how much to crop

Google Trends' most recent numbers are the least trustworthy — the latest period is sampled from incomplete data (and revised upward later), and recent interest is the most event-driven. So before fitting we **crop** the recent tail. To choose how much, we grid-searched the crop and validated by **forward holdout**: for each of 87 keywords, hold out a 26-week block ending at the crop boundary and predict it from a model trained only on the data *before* it. Recency bias shows up as poor predictability of recent blocks.

```
months cropped   held-out shape correlation       % keywords w/ skill
   0   ▮▮▮▮▮▮▮▮▮▮▮▮▮              0.28  last month IN      6%
   1   ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮  0.42  partial-period out 6%
   3   ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮     0.38                      8%
   6   ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮     0.38                      8%
  12   ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮     0.38  <- we crop here    17%
  18   ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮ 0.44                     28%
  24   ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮     0.38                     18%
```

Three things stand out:

- **The last month is the worst.** Cropping just one month lifts held-out shape correlation from **0.28 to 0.42** — the classic partial-period artefact, where the latest point is undercounted (in our data the final weeks sit ~25% below their local level).
- **Recency fades over about a year.** The share of keywords with any forward skill climbs from **6%** (no crop) to **28%** by ~18 months, then flattens. We crop **12 months** — enough to clear the volatile, revisable tail without discarding signal.
- **The model describes, it doesn't forecast.** Held-out $R^2$ stays negative throughout: this celestial fit explains the *historical* rhythm in-sample, it cannot predict the future. The crop simply keeps unreliable recent data from distorting that description.

One honest caveat: recent values also sit *above* their all-time seasonal average simply because search interest **grows** over time — a trend, not bias. The effects above are measured against the *local* level, which removes that trend. The full interactive chart is on the [recency page](https://artaquest.org/research/recency.html).

## What we verified (and what doesn't hold up)

I checked the machinery against the real ephemeris.

**Holds up.** The sinc kernel and bandwidths are exact (Moon half-width 4.3°, Sun 57.3°). The shares are non-negative and sum to $R^2$ across all 87 topics. The Sun's feature correlates strongly with a pure annual cycle — the strongest *genuinely varying* body is essentially encoding ordinary calendar seasonality.

**Doesn't hold up — read the high $R^2$ with care.**

- **The fast bodies don't matter — even now that we can see them.** Going weekly resolves them (Mercury ~13 samples per cycle, the Moon marginally at ~4) where monthly data aliased them — yet across the atlas the Moon and Mercury together carry only ≈0.01% of the fit. So their irrelevance isn't a sampling artefact; the fast-body cycles genuinely don't track what the world searches for.
- **Slow bodies aren't really cycles.** From Jupiter outward the sinc never completes even a quarter-bump in 22 years — those features are smooth ramps acting as a **low-frequency trend basis**, not 165-year rhythms we could ever observe cycling.
- So most of a high $R^2$ is a slow **multi-year trend** (outer bodies) plus annual **seasonality** (the Sun). Neither is evidence of a planetary cause.

**Correlation is not causation.** This is a statistical curiosity — it only measures whether two rhythms happen to line up over 22 years. Nothing here claims the sky *causes* anything, or that astrology is real.

The value is the transparency: twelve clocks, a sweep of angles, one MSE loss, and an honest non-negative decomposition of the result — so anyone can see, topic by topic, exactly how well a cosmic rhythm happens to line up with a human one.
