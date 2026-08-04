#!/usr/bin/env python3
"""Junk-word filter — keeps the atlas to REAL fields/styles, not Google-Trends function-word noise.

The raw 10k pool leaks stop-words ("there", "which", "does"), bare directions/quantifiers ("back", "last",
"best"), and meta terms ("video", "name", "season") that screen as "trending" (everyone searches them) but
are not fields of knowledge or descriptive styles. `is_junk()` is the single gate used everywhere:
  • screen_pool.py / build_funnel.py — never screen or finalise a junk word,
  • build_disciplines.py / export_research.py — never surface one even if it lingers in a registry,
  • purge_junk.py — strip existing junk from the pool + every analysis registry + the plugin data.

A word is junk if it is too short / non-alphabetic, a function word, or a generic non-topic term. Real but
generic content words (money, city, balance, drive, sport, music…) are deliberately KEPT.
"""

# Standard English stop-words (function words: articles, pronouns, prepositions, conjunctions, auxiliaries,
# common adverbs, question words) — the NLTK list, inlined so this stays dependency-free.
_STOP = """
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers
herself it its itself they them their theirs themselves what which who whom this that these those am is are
was were be been being have has had having do does did doing a an the and but if or because as until while of
at by for with about against between into through during before after above below to from up down in out on
off over under again further then once here there when where why how all any both each few more most other
some such no nor not only own same so than too very can will just should now also o re ve ll d m t s y ain
aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn
""".split()

# Generic / meta non-topic words: bare quantifiers, directions, time markers, ordinals, evaluations, and
# content-free meta terms. These screen as popular but name no field or style.
_GENERIC = """
best worst better worse first last latest next previous final initial new old recent former
back front top bottom left right side middle centre center end start begin around
free full empty half whole part total none some many much few several various
good bad great nice cool fine okay ok wrong correct true false real fake
today tomorrow yesterday now then soon late early time times date day days daily week weeks weekly
month months monthly year years yearly hour hours season seasons moment minute second
video audio photo image picture pic clip movie show name names title word words letter number numbers
order charge work works working job thing things stuff item items list lists type types kind kinds
form forms set sets group groups place places area areas spot point points line lines way ways
make made making get got getting go going gone come came want need use used using
how what why when where who which does done doing said say says
near far high low long short big small large huge tiny fast slow easy hard simple
hello world test sample example demo update news info data
never always ever often sometimes usually rarely seldom maybe perhaps indeed really quite rather
yes nay yeah okay sure done gone able unable enough almost nearly hardly mostly mainly simply merely
""".split()

STOPWORDS = set(_STOP) | set(_GENERIC)


def is_junk(word: str) -> bool:
    w = (word or "").strip().lower()
    if len(w) < 3:                              # 1–2 letter tokens are never a field
        return True
    if any(c.isdigit() for c in w):             # numbers / codes
        return True
    if not all(c.isalpha() or c in " -'" for c in w):  # punctuation/symbols (but keep hyphens, spaces, apostrophes)
        return True
    if " " in w or "-" in w:                    # compound terms ("anthropology-sociology", "sci-fi") are real fields
        return False
    return w in STOPWORDS                       # a bare single token is junk only if it's a stop/generic word


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            print(f"{a}: {'JUNK' if is_junk(a) else 'keep'}")
    else:
        print(f"{len(STOPWORDS)} stop/junk words")
