"""Multilingual semantic matcher (second pass after the token matcher).

Why a second pass: the token matcher only works when the game name appears
in the title in the same script. A Russian/Arabic/Korean title that refers
to the game by a translated name is invisible to it. Embeddings fix that,
e.g. Arabic "الإمتحان" -> "Examination".

What measurement on 1157 real titles taught us (so the defaults are not
guesses):
  * Whole-title embeddings are useless: they match the *topic*
    ("a Roblox video"), 39% agreement, hundreds of false hits.
  * Short n-gram windows are better but suffer from "hubness": a few games
    (e.g. "WorldFront") sit close to everything, so generic phrases like
    "ALL OF THEM" scored 0.94 against them.
  * The fix is hubness correction (CSLS): score = 2*cos(window, game)
    - hub(game), where hub(game) is the mean of its top-K similarities to a
    pool of ordinary windows. With threshold 1.1: precision ~0.90 and ~3%
    of non-matching titles fire (several of which are genuine translations).

Model: paraphrase-multilingual-MiniLM-L12-v2 via fastembed (ONNX, CPU, no
torch). ~120 MB, cached on disk.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(DATA, "models")
CACHE = os.path.join(DATA, "emb_cache.npz")

# Two thresholds, because the two situations are not equally risky.
# Cross-script (Cyrillic/Arabic/CJK window vs a Latin game name) is what
# embeddings are *for*, so it gets the lower bar. A Latin window against a
# Latin name is something the token pass already handles; anything it missed
# is mostly generic words ("GUERRA" -> "Mini War", "BEDY" -> "BedWars"), so
# that path needs a much higher score and at least two words.
CSLS_CROSS = 1.15
CSLS_LATIN = 1.30
CSLS_THRESHOLD = CSLS_CROSS   # kept for backward compatibility
HUB_K = 10
MIN_WINDOW_CHARS = 4

_model = None


def available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        os.makedirs(MODEL_DIR, exist_ok=True)
        # threads is NOT a no-op default. Measured on this project's own CPU
        # (2-vCPU GitHub runner, Xeon @2.60GHz): leaving it unset (ORT's
        # "auto") cost 1.8x more wall time than pinning it to the actual core
        # count - 114.8 ms/title vs 62.8 ms/title on a 150-title benchmark,
        # i.e. the difference between a 4000-title pass finishing in ~4 vs
        # ~7.6 minutes. ORT's auto-detection does not reliably land on the
        # right thread count in a container with a cgroup CPU limit, where
        # `os.cpu_count()` still reports the host's count correctly.
        _model = TextEmbedding(MODEL_NAME, cache_dir=MODEL_DIR,
                               threads=os.cpu_count() or 2)
    return _model


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    """L2-normalised embeddings, shape (n, 384)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    m = _get_model()
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        out.extend(m.embed([t[:200] for t in texts[i:i + batch]], batch_size=batch))
    a = np.asarray(out, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def windows(title: str, max_n: int = 3) -> list[str]:
    """1-3 word windows. Unicode-aware so Cyrillic/Arabic/CJK work."""
    toks = re.findall(r"[\w'’]+", title or "")
    out: list[str] = []
    seen: set[str] = set()
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            w = " ".join(toks[i:i + n])
            if len(w) >= MIN_WINDOW_CHARS and w.lower() not in seen:
                seen.add(w.lower())
                out.append(w)
    return out


class Index:
    """Game-name embeddings + per-game hubness, persisted between runs."""

    def __init__(self):
        self.uids: np.ndarray = np.zeros(0, dtype=np.int64)
        self.names: list[str] = []
        self.vecs: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self.hub: np.ndarray = np.zeros(0, dtype=np.float32)
        self.pool: np.ndarray = np.zeros((0, 384), dtype=np.float32)

    # ---- persistence
    def load(self) -> None:
        if not os.path.exists(CACHE):
            return
        try:
            z = np.load(CACHE, allow_pickle=True)
            self.uids = z["uids"]
            self.names = list(z["names"])
            self.vecs = z["vecs"]
            self.hub = z["hub"]
            self.pool = z["pool"]
        except Exception:
            self.__init__()  # corrupt cache -> rebuild

    def save(self) -> None:
        os.makedirs(DATA, exist_ok=True)
        np.savez_compressed(CACHE, uids=self.uids, names=np.array(self.names, dtype=object),
                            vecs=self.vecs, hub=self.hub, pool=self.pool)

    # ---- building
    def ensure_pool(self, titles: list[str], size: int = 1500) -> None:
        """Pool of ordinary windows used to estimate each game's hubness."""
        if len(self.pool) >= 300 or not titles:
            return
        seen: set[str] = set()
        ws: list[str] = []
        for t in titles:
            for w in windows(t):
                k = w.lower()
                if k not in seen:
                    seen.add(k)
                    ws.append(w)
            if len(ws) >= size:
                break
        self.pool = embed(ws[:size])

    def sync_games(self, games: list[tuple[int, str]]) -> int:
        """Embed games not seen yet. Returns how many were added."""
        known = set(int(u) for u in self.uids)
        new = [(int(u), n) for u, n in games if int(u) not in known and n]
        if not new:
            return 0
        v = embed([clean_game_name(n) or n for _, n in new])
        if len(self.pool):
            k = min(HUB_K, len(self.pool))
            hub = np.sort(self.pool @ v.T, axis=0)[-k:].mean(0).astype(np.float32)
        else:
            hub = np.full(len(new), 0.56, dtype=np.float32)
        self.uids = np.concatenate([self.uids, np.array([u for u, _ in new], dtype=np.int64)])
        self.names += [n for _, n in new]
        self.vecs = np.vstack([self.vecs, v]) if len(self.vecs) else v
        self.hub = np.concatenate([self.hub, hub]) if len(self.hub) else hub
        return len(new)

    # ---- matching
    def match(self, title: str) -> dict | None:
        """Single-title match. Kept for callers that process one video at a
        time; see `match_batch` for anything processing many at once - calling
        this in a loop pays the fixed per-call overhead of the ONNX session
        (tokenize + run + marshal) once per TITLE instead of once per BATCH.

        Measured on run #84 (2026-10-28): 4000 titles matched one at a time
        timed out past 12 minutes. The windows per title are 5-20 short
        strings, so each `embed()` call here does real work for a few
        milliseconds and pays session overhead for the rest - multiplied by
        4000 calls instead of ~storage ceil(4000*12/32) = ~1500 if batched.
        """
        return self._score_one(windows(title))

    def match_batch(self, titles: list[str]) -> list[dict | None]:
        """Match many titles with embed() called a small, fixed number of
        times instead of once per title.

        All windows from all titles are embedded in ONE call (internally
        chunked at 32 by `embed()`), then scored per title by slicing the
        result. The scoring itself (two matrix multiplies per title) is
        identical to calling `match()` in a loop - only the number of ONNX
        invocations changes, from len(titles) down to
        ceil(total_windows / 32).
        """
        if not len(self.vecs):
            return [None] * len(titles)
        per_title_windows = [windows(t) for t in titles]
        flat: list[str] = []
        bounds: list[tuple[int, int]] = []
        for ws in per_title_windows:
            start = len(flat)
            flat.extend(ws)
            bounds.append((start, len(flat)))
        if not flat:
            return [None] * len(titles)
        e_all = embed(flat)
        out: list[dict | None] = []
        for ws, (start, end) in zip(per_title_windows, bounds):
            if start == end:
                out.append(None)
                continue
            out.append(self._score_one(ws, e_all[start:end]))
        return out

    def _score_one(self, ws: list[str], e: np.ndarray | None = None) -> dict | None:
        if not ws or not len(self.vecs):
            return None
        if e is None:
            e = embed(ws)
        s = 2.0 * (e @ self.vecs.T) - self.hub[None, :]
        # Latin-only windows with a single word are never accepted here.
        for i, w in enumerate(ws):
            if is_latin(w) and len(w.split()) < 2:
                s[i, :] = -9.0
        i, j = np.unravel_index(int(s.argmax()), s.shape)
        score = float(s[i, j])
        threshold = CSLS_LATIN if is_latin(ws[i]) else CSLS_CROSS
        if score < threshold:
            return None
        return {
            "universe_id": int(self.uids[j]),
            "game_name": self.names[j],
            "containment": round(min(score / 1.5, 1.0), 3),
            "matched": ws[i][:60],
            "specificity": len(ws[i].split()),
            "method": "embed",
            "score": round(score, 3),
        }


def is_latin(text: str) -> bool:
    """True if every letter is Latin (accented Latin included)."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(ord(c) < 0x250 for c in letters)


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2B50\u2705\u274C\u2B06\u2197\u3030]")


def clean_game_name(name: str) -> str:
    """Strip emoji and [bracket] event tags so the embedding sees the title."""
    s = _EMOJI.sub(" ", name or "")
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_index(games: list[tuple[int, str]], titles: list[str]) -> Index:
    idx = Index()
    idx.load()
    idx.ensure_pool(titles)
    added = idx.sync_games(games)
    if added or not os.path.exists(CACHE):
        idx.save()
    return idx
