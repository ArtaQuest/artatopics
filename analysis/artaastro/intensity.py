#!/usr/bin/env python3
"""
intensity.py — ArtaAstro's A-PRIORI mundane "event-intensity" model.

This file encodes classical mundane-astrology heuristics as a fixed, prespecified rule set that
maps a single day's sidereal-Lahiri sky to a 0-100 "event-intensity" score. THE WEIGHTS HERE ARE
FROZEN BEFORE THE GDELT GROUND TRUTH IS EVER TOUCHED. They are never fitted, tuned, or selected
against the backtest data — doing so would turn an honest out-of-sample test into circular
overfitting. (This is the whole point of the exercise: test whether a *pre-committed* astrological
theory has any skill, with a permutation null. The expected, and honestly reported, answer may well
be "no better than chance.")

Astrological rationale (standard mundane astrology, e.g. Ptolemy/Barbault/Green lineage):
  * Hard aspects (0/90/180) between the slow "social/transpersonal" bodies mark eras of tension and
    upheaval; Saturn-Pluto in particular is the classic "hard times / power crisis" cycle.
  * Eclipses are traditionally omens, more so when they fall on a malefic (Mars/Saturn) or the nodes.
  * A slow planet changing sign (ingress) is a mundane "regime shift".
  * A planet stationing (turning ret/direct) is a pressure/pivot point.
Only the SLOW bodies + Mars carry mundane weight here (Moon/Mercury/Venus are personal & fast).

The score is deliberately simple and transparent so the null test is meaningful.
"""
import math

# ---- geometry -------------------------------------------------------------------------------------
ASPECTS = {"conjunction": 0.0, "sextile": 60.0, "square": 90.0, "trine": 120.0, "opposition": 180.0}
HARD    = {"conjunction", "square", "opposition"}
ORB     = 3.0                      # tight orb (deg); slow bodies, so 3 deg is already several weeks

# bodies that carry mundane weight for pair-aspects
MUNDANE = ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Rahu"]

# base weight per unordered pair (frozenset). Saturn-Pluto highest, per the mundane tradition.
def _pw(a, b, w): return (frozenset((a, b)), w)
PAIR_WEIGHT = dict([
    _pw("Saturn", "Pluto",   10.0),
    _pw("Saturn", "Uranus",   8.0),
    _pw("Uranus", "Pluto",    7.0),
    _pw("Jupiter","Saturn",   7.0),   # the ~20-yr Great Conjunction
    _pw("Saturn", "Neptune",  6.0),
    _pw("Jupiter","Uranus",   5.0),
    _pw("Jupiter","Pluto",    5.0),
    _pw("Mars",   "Saturn",   6.0),
    _pw("Mars",   "Pluto",    6.0),
    _pw("Mars",   "Uranus",   5.0),
    _pw("Rahu",   "Saturn",   5.0),
    _pw("Rahu",   "Mars",     5.0),
    _pw("Rahu",   "Pluto",    4.0),
    _pw("Neptune","Pluto",    4.0),
])
TYPE_FACTOR = {"conjunction": 1.0, "opposition": 1.0, "square": 0.85, "trine": 0.35, "sextile": 0.25}

# non-aspect contributions
ECLIPSE_W   = {"total": 8.0, "annular": 5.0, "partial": 5.0}
MALEFIC     = ("Mars", "Saturn", "Rahu", "Ketu")
INGRESS_W   = {"Saturn": 4.0, "Rahu": 4.0, "Jupiter": 3.5, "Pluto": 3.0, "Uranus": 3.0, "Neptune": 3.0}
STATION_W   = {"Mars": 3.0, "Saturn": 3.0, "Uranus": 2.5, "Neptune": 2.5, "Pluto": 2.5, "Jupiter": 2.0}

SATURATION_K = 20.0   # score = 100*(1-exp(-raw/K)); FIXED a priori. ~raw 20 -> 63, ~40 -> 86.


def _sep(a, b):
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180.0 else d


def aspects_between(bodies):
    """List of (bodyA, bodyB, aspect_name, orb_distance) for MUNDANE bodies within ORB."""
    out = []
    names = [n for n in MUNDANE if n in bodies]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sep = _sep(bodies[a]["lon"], bodies[b]["lon"])
            for name, ang in ASPECTS.items():
                d = abs(sep - ang)
                if d <= ORB:
                    out.append((a, b, name, d))
                    break
    return out


def score(day):
    """
    day: {
      'bodies':  {name: {'lon','speed','retro','sign'}},
      'ingress': [slow body names that changed sign today],
      'station': [outer body names turning ret/direct today],
      'eclipse': None | {'kind': 'total'|'annular'|'partial', 'lon': float, 'lunar': bool},
    }
    returns (score_0_100, raw, triggers[list[str]])
    """
    bodies = day["bodies"]
    raw = 0.0
    triggers = []

    # 1) slow-pair aspects (orb-tapered)
    for a, b, name, d in aspects_between(bodies):
        w = PAIR_WEIGHT.get(frozenset((a, b)))
        if not w:
            continue
        contrib = w * TYPE_FACTOR[name] * (1.0 - d / ORB)
        if contrib <= 0:
            continue
        raw += contrib
        if name in HARD and contrib >= 1.0:
            triggers.append(f"{a}–{b} {name} (orb {d:.1f}°)")

    # 2) eclipse (build.py passes the eclipse within its +-3 day window, if any)
    ec = day.get("eclipse")
    if ec:
        w = ECLIPSE_W.get(ec.get("kind"), 5.0)
        # boost if the eclipse degree sits on a malefic / node
        boost = 1.0
        elon = ec.get("lon")
        if elon is not None:
            for m in MALEFIC:
                if m in bodies and _sep(elon, bodies[m]["lon"]) <= 5.0:
                    boost = 1.5
                    break
        raw += w * boost
        triggers.append(("Lunar" if ec.get("lunar") else "Solar") + f" {ec.get('kind')} eclipse"
                        + (" on a malefic" if boost > 1 else ""))

    # 3) ingress of a slow body
    for nm in day.get("ingress", []):
        w = INGRESS_W.get(nm)
        if w:
            raw += w
            triggers.append(f"{nm} ingress ({bodies[nm]['sign'] if nm in bodies else '?'})")

    # 4) outer-planet station
    for nm in day.get("station", []):
        w = STATION_W.get(nm)
        if w:
            raw += w
            triggers.append(f"{nm} station")

    s = 100.0 * (1.0 - math.exp(-raw / SATURATION_K))
    return s, raw, triggers
