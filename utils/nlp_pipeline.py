"""
NLP Preprocessing Pipeline
- Tokenisation
- Intent Classification
- Entity Extraction
Reduces query failure rate by ~35% (as per project spec)
"""

import re
from typing import Dict, Any

# Try spaCy; fall back to regex-only if model not installed
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    _nlp = None
    SPACY_AVAILABLE = False

# Try NLTK stopwords
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    _STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "for"}


# ─── Intent taxonomy ────────────────────────────────────────────────────────

INTENT_PATTERNS: Dict[str, list] = {
    "summary":      [r"\bsummar", r"\boverview", r"\bdescribe", r"\bwhat is in", r"\btell me about"],
    "trend":        [r"\btrend", r"\bover time", r"\bchange", r"\bgrowth", r"\bdecline", r"\bincrease", r"\bdecrease"],
    "outlier":      [r"\boutlier", r"\banomal", r"\bspike", r"\bunusual", r"\bextreme"],
    "correlation":  [r"\bcorrelat", r"\brelation", r"\bconnect", r"\bdepend", r"\blink"],
    "top_n":        [r"\btop\s*\d*", r"\bbest", r"\bhighest", r"\bmaximum", r"\bmost"],
    "bottom_n":     [r"\bbottom\s*\d*", r"\bworst", r"\blowest", r"\bminimum", r"\bleast"],
    "distribution": [r"\bdistribut", r"\bhistogram", r"\bspread", r"\brange", r"\bfrequency"],
    "filter":       [r"\bwhere\b", r"\bfilter", r"\bonly\b", r"\bshow.*where", r"\bfor\s+\w+\s*="],
    "aggregate":    [r"\btotal", r"\bsum\b", r"\baverage", r"\bmean\b", r"\bcount\b", r"\bgroup by"],
    "recommend":    [r"\brecommend", r"\bsuggestion", r"\badvice", r"\bshould\b", r"\boptimis"],
    "compare":      [r"\bcompar", r"\bvs\b", r"\bversus", r"\bdifference", r"\bbetween"],
    "forecast":     [r"\bforecast", r"\bpredict", r"\bfuture", r"\bnext"],
}


class NLPPipeline:
    """Pre-processes natural-language queries before sending to the LLM."""

    # ── Public API ──────────────────────────────────────────────────────────

    def process(self, query: str) -> Dict[str, Any]:
        """
        Returns:
            {
              "original":   str,
              "clean":      str,
              "tokens":     list[str],
              "intent":     str,
              "entities":   list[dict],
              "keywords":   list[str],
              "numbers":    list[float],
              "confidence": float,
            }
        """
        clean = self._clean(query)
        tokens = self._tokenise(clean)
        intent, confidence = self._classify_intent(query)
        entities = self._extract_entities(query, tokens)
        keywords = self._extract_keywords(tokens)
        numbers = self._extract_numbers(query)

        return {
            "original": query,
            "clean": clean,
            "tokens": tokens,
            "intent": intent,
            "entities": entities,
            "keywords": keywords,
            "numbers": numbers,
            "confidence": confidence,
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenise(self, text: str) -> list:
        if SPACY_AVAILABLE and _nlp:
            doc = _nlp(text)
            return [t.lemma_ for t in doc if not t.is_space]
        # Fallback: simple whitespace split
        return re.findall(r"\b\w+\b", text)

    def _classify_intent(self, query: str) -> tuple:
        q = query.lower()
        scores: Dict[str, int] = {}
        for intent, patterns in INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, q))
            if score:
                scores[intent] = score

        if not scores:
            return "general", 0.5

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = round(scores[best] / max(total, 1), 2)
        return best, confidence

    def _extract_entities(self, query: str, tokens: list) -> list:
        entities = []

        # Use spaCy NER when available
        if SPACY_AVAILABLE and _nlp:
            doc = _nlp(query)
            for ent in doc.ents:
                entities.append({"text": ent.text, "label": ent.label_})

        # Regex-based column-name hints (words in quotes or after "column"/"field")
        col_hints = re.findall(r'["\']([\w\s]+)["\']|column\s+(\w+)|field\s+(\w+)', query, re.I)
        for groups in col_hints:
            hint = next((g for g in groups if g), None)
            if hint:
                entities.append({"text": hint.strip(), "label": "COLUMN_HINT"})

        # Detect "top N" numbers
        top_n = re.search(r"top\s*(\d+)", query, re.I)
        if top_n:
            entities.append({"text": top_n.group(1), "label": "TOP_N"})

        return entities

    def _extract_keywords(self, tokens: list) -> list:
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 2 and t.isalpha()]

    def _extract_numbers(self, query: str) -> list:
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", query)
        return [float(n) for n in nums]


# ─── Quick smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipe = NLPPipeline()
    test_queries = [
        "What are the top 5 routes by revenue?",
        "Show me outliers in ticket_price column",
        "How does occupancy trend over time?",
        "Compare domestic vs international flights",
        "Give recommendations to optimise revenue",
    ]
    for q in test_queries:
        result = pipe.process(q)
        print(f"\nQuery   : {q}")
        print(f"Intent  : {result['intent']} (conf={result['confidence']})")
        print(f"Keywords: {result['keywords']}")
        print(f"Entities: {result['entities']}")
