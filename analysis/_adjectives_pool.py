#!/usr/bin/env python3
"""The ADJECTIVE discovery vocabulary — the HOW camp of every house.

Each house is split into two camps (operator 2026-06-27): WHAT (nouns — fields of knowledge) and HOW
(adjectives — the style/quality of the content). The noun pool is analysis/_sign_themes_pool.py (singular
nouns); THIS file is the parallel adjective pool. Every word here is a real, common English descriptive
adjective that reads naturally in front of a field ("Epic History", "Practical Economics", "Ancient Art").
The funnel screens + sidereally fits these exactly like nouns; a fitted adjective lands in a house and becomes
a candidate HOW (adjective) representative. `pos` is decided by membership in this set (see build_disciplines.py).

Kept deliberately free of stop-words and value-laden/junk terms ("best", "last", "right") — those make ugly
recommendations and leak from the raw Trends pool; this is the curated, intentional adjective surface.
"""
ADJECTIVES = [
    "advanced", "ancient", "applied", "artistic", "atomic", "basic", "bold", "brave", "bright", "brilliant",
    "calm", "classic", "classical", "clever", "colourful", "complete", "complex", "cosmic", "creative", "critical",
    "cultural", "curious", "daring", "deep", "delicate", "digital", "dramatic", "dynamic", "easy", "elegant",
    "electric", "elemental", "엔", "epic", "essential", "eternal", "ethical", "exotic", "experimental", "expert",
    "extreme", "famous", "fast", "fearless", "fertile", "fierce", "fine", "fluid", "formal", "foundational",
    "fundamental", "futuristic", "gentle", "glamorous", "global", "golden", "gothic", "graceful", "grand", "gritty",
    "heroic", "historic", "holistic", "humble", "iconic", "imaginative", "immersive", "industrial", "infinite", "innovative",
    "inspiring", "intense", "intensive", "interactive", "intermediate", "intricate", "intuitive", "legendary", "logical", "luminous",
    "magical", "majestic", "masterful", "mathematical", "mechanical", "medieval", "mental", "meticulous", "military", "minimal",
    "modern", "molecular", "musical", "mystical", "natural", "noble", "nuclear", "organic", "original", "ornate",
    "philosophical", "physical", "playful", "poetic", "political", "practical", "precise", "premium", "primal", "profound",
    "quantum", "quiet", "radical", "rapid", "rare", "rational", "raw", "realistic", "rebellious", "refined",
    "regal", "rhythmic", "rich", "romantic", "rugged", "rural", "sacred", "scientific", "sculptural", "serene",
    "sharp", "simple", "skilful", "smart", "social", "solar", "sonic", "spatial", "spiritual", "strategic",
    "structural", "subtle", "surreal", "sustainable", "swift", "symbolic", "systematic", "tactical", "technical", "tender",
    "theatrical", "theoretical", "thoughtful", "timeless", "traditional", "tranquil", "tribal", "tropical", "ultimate", "universal",
    "urban", "vibrant", "vintage", "viral", "virtual", "visceral", "visual", "vivid", "vocal", "wild",
]
# Sanitise: drop any accidental non-ASCII entry, dedupe, lowercase.
ADJECTIVES = sorted({w.strip().lower() for w in ADJECTIVES if w.isascii() and w.isalpha()})
ADJ_SET = set(ADJECTIVES)

if __name__ == "__main__":
    print(len(ADJECTIVES), "adjectives")
    print(" ".join(ADJECTIVES))
