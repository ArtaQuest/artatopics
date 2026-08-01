"""IMPORT SHIM — replaces the artaquest monorepo module of the same name.

In the monorepo, analysis/adstopics/astro_phasor2.py belongs to a different research campaign and
chains through two more modules and that campaign's own data files. The topics campaign only ever
reads two constant lists from it (BODIES, SIGNS — verified by census: no other attribute is touched),
so this standalone repository carries just those constants, at the same path, and every script runs
unchanged.
"""

BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn",
          "uranus", "neptune", "pluto", "node", "chiron"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
