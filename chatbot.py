# chatbot.py
"""Sentiment-aware chatbot (VADER + canned responses from responses.csv).

IMPORTANT: this is NOT a medical or crisis tool. VADER only scores the tone of
a sentence; it cannot reliably detect that someone is in danger. A small list
of crisis phrases triggers a static supportive message, but that list is
limited (English only) and can miss things.
"""
import csv
import logging
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESPONSES_FILE = BASE_DIR / "responses.csv"

logger = logging.getLogger(__name__)

SENTIMENTS = ("positive", "neutral", "negative")

# Used only if a pool is somehow empty at runtime.
FALLBACK_RESPONSES = {
    "positive": "That's great to hear!",
    "neutral": "I see. Tell me more if you'd like.",
    "negative": "I'm sorry you're going through this. I'm here to listen.",
}

# English phrases that switch the bot to a supportive static message.
CRISIS_PHRASES = (
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "want to die",
    "wanna die",
    "don't want to live",
    "do not want to live",
    "hurt myself",
    "harm myself",
    "self harm",
    "self-harm",
)

# No phone numbers on purpose: add a helpline for YOUR country after checking
# it on an official source.
CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this much pain. You deserve support from a real person "
    "right now. If you might be in danger, please contact your local emergency number, or reach "
    "out to a trusted friend, family member or a mental health professional. "
    "I'm a simple wellness chatbot and can't replace professional help."
)


def load_response_pools(filename=DEFAULT_RESPONSES_FILE):
    """Read the responses CSV into {"positive": [...], "neutral": [...], "negative": [...]}.

    Raises ValueError for malformed rows (e.g. an unquoted comma inside a
    response) or if a sentiment has no responses at all.
    """
    pools = {name: [] for name in SENTIMENTS}
    with open(filename, mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or {"Sentiment", "Response"} - set(reader.fieldnames):
            raise ValueError("responses file must have the columns: Sentiment, Response")
        for row in reader:
            if None in row or row.get("Response") is None:
                raise ValueError(
                    f"Malformed row near line {reader.line_num} in {filename}: "
                    "put responses that contain a comma inside double quotes."
                )
            sentiment = (row["Sentiment"] or "").strip().lower()
            response = row["Response"].strip()
            if sentiment in pools and response:
                pools[sentiment].append(response)

    empty = [name for name, items in pools.items() if not items]
    if empty:
        raise ValueError(f"No responses found for sentiment(s): {', '.join(empty)}")
    return pools


def create_analyzer():
    """Create a VADER analyzer (imported here so other modules work without it)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def is_crisis_message(text):
    lowered = " ".join(str(text).lower().replace("’", "'").split())
    return any(phrase in lowered for phrase in CRISIS_PHRASES)


def classify_sentiment(text, analyzer):
    """Return ("positive" | "neutral" | "negative", compound_score)."""
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive", score
    if score <= -0.05:
        return "negative", score
    return "neutral", score


def build_reply(user_input, analyzer, pools):
    """Return {"response", "sentiment", "crisis"} for the user's message."""
    if is_crisis_message(user_input):
        logger.info("Chatbot detected sentiment: crisis")
        return {"response": CRISIS_RESPONSE, "sentiment": "crisis", "crisis": True}

    sentiment, _ = classify_sentiment(user_input, analyzer)
    pool = pools.get(sentiment) or [FALLBACK_RESPONSES[sentiment]]
    logger.info("Chatbot detected sentiment: %s", sentiment)
    return {"response": random.choice(pool), "sentiment": sentiment, "crisis": False}


# ---- Simple module-level API (kept for the command-line usage) ----
_analyzer = None
_pools = None


def handle_chatbot_request(user_input):
    """Return the chatbot's text reply (loads VADER and the CSV on first use)."""
    global _analyzer, _pools
    if _analyzer is None:
        _analyzer = create_analyzer()
    if _pools is None:
        _pools = load_response_pools()
    return build_reply(user_input, _analyzer, _pools)["response"]


# Backwards-compatible names used by the old code.
generate_response = handle_chatbot_request


def analyze_sentiment(text):
    """Return the VADER compound score (-1 to +1) for `text`."""
    global _analyzer
    if _analyzer is None:
        _analyzer = create_analyzer()
    return _analyzer.polarity_scores(text)["compound"]


if __name__ == "__main__":
    logging.basicConfig(
        filename=BASE_DIR / "app.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending conversation. Goodbye!")
            break
        print(f"Chatbot: {handle_chatbot_request(user_input)}")
