"""Tiny stand-ins for optional third-party packages.

They are installed ONLY when the real package cannot be imported (for example
in an offline sandbox). On a normal machine with `pip install -r requirements.txt`
the real flask-cors and vaderSentiment are used and nothing is stubbed.
"""
import sys
import types

STUBBED = []


def install_stubs():
    try:
        import flask_cors  # noqa: F401
    except ImportError:
        module = types.ModuleType("flask_cors")
        module.CORS = lambda app, **kwargs: app
        sys.modules["flask_cors"] = module
        STUBBED.append("flask_cors")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: F401
    except ImportError:
        positive = {"great", "happy", "good", "awesome", "love", "wonderful", "amazing"}
        negative = {"sad", "terrible", "awful", "bad", "hate", "lonely", "depressed", "horrible"}

        class SentimentIntensityAnalyzer:  # crude word-list scorer, NOT real VADER
            def polarity_scores(self, text):
                words = [w.strip(".,!?").lower() for w in text.split()]
                score = sum(w in positive for w in words) - sum(w in negative for w in words)
                return {"compound": max(-1.0, min(1.0, score * 0.4))}

        package = types.ModuleType("vaderSentiment")
        inner = types.ModuleType("vaderSentiment.vaderSentiment")
        inner.SentimentIntensityAnalyzer = SentimentIntensityAnalyzer
        package.vaderSentiment = inner
        sys.modules["vaderSentiment"] = package
        sys.modules["vaderSentiment.vaderSentiment"] = inner
        STUBBED.append("vaderSentiment")
