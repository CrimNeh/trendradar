"""Niche classification from name + description.

Roblox's own `genre` field is now always "All", so niche must be derived.
A single regex hit is too noisy, so hits are weighted by position (name 3x,
description opening 2x, body 1x) and a minimum score is required.
"""
from __future__ import annotations

import re

NICHE_RULES: dict[str, str] = {
    "obby": r"\bobby\b|\bobstacle course\b|\bparkour\b",
    "tycoon": r"\btycoon\b|\bfactory\b|\bempire\b|\bmanager\b|\bmoney machine\b|\bproduction\b",
    "simulator": r"\bsimulator\b|\bclicker\b|\bincremental\b|\bidle\b",
    "tower_defense": r"\btower defen[cs]e\b|\btd\b",
    "battle_royale": r"\bbattle ?royale\b|\blast (one|man|person) standing\b|\bsurvival games?\b",
    "shooter": r"\bfps\b|\bshooter\b|\bdeathmatch\b|\btdm\b|\bsniper\b|\baim trainer\b",
    "fighting": r"\bboxing\b|\bfighting\b|\bbrawl\b|\bmma\b|\bwrestl\w*\b",
    "horror": r"\bhorror\b|\bscary\b|\bcreepy\b|\bbackrooms\b|\bjumpscare\b|\bhaunted\b",
    "rpg": r"\brpg\b|\bmmorpg\b|\bquest[s]?\b|\bdungeon\b|\bfantasy\b|\bclasses\b",
    "roleplay": r"\broleplay\b|\bhangout\b|\bcity life\b|\bapartment\b|\bcafe\b|\bhigh school\b",
    "racing": r"\bracing\b|\bdrift\w*\b|\bkart\b|\brace ?(track|car)\b",
    "sports": r"\bsoccer\b|\bfootball\b|\bbasketball\b|\btennis\b|\bgolf\b|\bhockey\b",
    "building": r"\bsandbox\b|\bcreative\b|\bbuild(ing)? (mode|your)\b|\bblueprint\b",
    "survival": r"\bsurvival\b|\bsurvive\b|\bcraft\w*\b|\bbase build\w*\b",
    "puzzle": r"\bpuzzle[s]?\b|\bquiz\b|\btrivia\b|\bguess the\b|\bfind the\b|\bmaze\b",
    "social_deduction": r"\bimpostor\b|\bmafia\b|\bdeduction\b|\bwho is the\b|\bsecret role\b",
    "fashion_beauty": r"\bmakeup\b|\bfashion\b|\bbeauty\b|\bmakeover\b|\boutfit\b|\bface ?card\b",
    "pets_animals": r"\bhatch\w*\b|\bpets?\b|\banimals?\b|\bcreatures?\b|\begg[s]?\b",
    "arena": r"\barena\b|\bpvp\b|\bduel\b|\b1v1\b|\btournament\b",
    "climbing": r"\bclimb\w*\b|\bascend\w*\b|\btower[s]?\b",
    "story": r"\bstory\b|\bchapter[s]?\b|\bjourney\b|\badventure\b",
    "stealth": r"\bsteal\w*\b|\bescape\b|\bheist\b|\bsneak\w*\b|\bthief\b",
    "farm_garden": r"\bfarm\w*\b|\bgarden\w*\b|\bharvest\b|\bplant[s]?\b|\bgrow\w*\b",
    "horror_multi": r"\bmultiplayer horror\b|\bco-?op horror\b",
    "dress_up": r"\bdress ?up\b|\baesthetic\b|\bvibe\b",
    "card_battle": r"\bcard[s]?\b|\bdeck\b|\btrading card\b",
}

MECHANIC_RULES: dict[str, str] = {
    "pvp": r"\bpvp\b|\bplayer vs\.? player\b|\bversus\b|\bcombat\b",
    "pve": r"\bpve\b|\bwaves?\b|\bzombies?\b|\benemies\b|\bboss(es)?\b",
    "gacha": r"\bgacha\b|\bcrates?\b|\bcases?\b|\bspin\b|\broll\b|\bloot\b|\bhatch\w*\b|\begg[s]?\b",
    "progression": r"\blevel\w*\b|\bxp\b|\brebirth\w*\b|\bprestige\b|\bupgrade\w*\b|\btier[s]?\b",
    "trading": r"\btrad(e|ing)\b|\bmarket\b|\bauction\b|\bvalues?\b",
    "coop": r"\bco-?op\b|\bcooperative\b|\bteam\w*\b|\bfriends\b",
    "customization": r"\bavatar\b|\bcustomi[sz]\w*\b|\bskins?\b|\bhair\b|\boutfit\b",
    "rebirth": r"\brebirth\w*\b|\bprestige\b",
    "codes": r"\bcodes?\b",
    "grind": r"\bgrind\w*\b|\bafk\b|\bidle\b|\bauto ?farm\b",
    "p2w": r"\bgamepass\b|\bpremium\b|\brobux\b|\bvip\b",
    "raid": r"\braid\w*\b|\bloot\b|\bextract\w*\b",
}

_NICHE = {k: re.compile(v, re.I) for k, v in NICHE_RULES.items()}
_MECH = {k: re.compile(v, re.I) for k, v in MECHANIC_RULES.items()}

MIN_SCORE = 3


def _score(rx: re.Pattern, name: str, head: str, body: str) -> int:
    return (3 * min(len(rx.findall(name)), 2)
            + 2 * min(len(rx.findall(head)), 2)
            + 1 * min(len(rx.findall(body)), 3))


def classify(name: str | None, description: str | None) -> tuple[str, str]:
    name, desc = name or "", description or ""
    head, body = desc[:200], desc[200:]
    scores = {k: _score(rx, name, head, body) for k, rx in _NICHE.items()}
    best = max(scores, key=scores.get)
    niche = best if scores[best] >= MIN_SCORE else "other"
    mechs = [k for k, rx in _MECH.items() if _score(rx, name, head, body) >= 2]
    return niche, ",".join(mechs)


def clean_name(raw: str | None) -> str:
    if not raw:
        return ""
    s = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2B50\u2705\u274C\u2B06\u2197]", "", raw)
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()
