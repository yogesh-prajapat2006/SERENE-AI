# app.py
"""SERENE-AI Flask API.

Run locally:      python app.py
Production:       gunicorn app:app        (Linux)
All error responses are JSON: {"error": "message"}.
"""
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

import ai_writing_therapist
import chatbot
import habit_clustering
import habit_recommendation
import music_therapy

BASE_DIR = Path(__file__).resolve().parent
MAX_CHATBOT_INPUT_CHARS = 1000


class ApiError(Exception):
    """An error that is shown to the API user as JSON."""

    def __init__(self, message, status=400, **extra):
        super().__init__(message)
        self.message = message
        self.status = status
        self.extra = extra


def get_json_object():
    """Return the JSON body as a dict or raise a 400 ApiError."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object (Content-Type: application/json).")
    return data


def require_text(data, key, max_chars=200):
    value = data.get(key)
    if value is None:
        raise ApiError(f"'{key}' is required.")
    if not isinstance(value, str) or not value.strip():
        raise ApiError(f"'{key}' must be a non-empty text value.")
    if len(value) > max_chars:
        raise ApiError(f"'{key}' is too long (maximum {max_chars} characters).")
    return value.strip()


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        RESPONSES_FILE=BASE_DIR / "responses.csv",
        HABITS_FILE=BASE_DIR / "habits_data.csv",
        MUSIC_FILE=BASE_DIR / "music_data.csv",
        MUSIC_DIR=BASE_DIR / "music",
        # Comma separated list of allowed frontend origins; "*" = everyone.
        CORS_ORIGINS=os.environ.get("CORS_ORIGINS", "*"),
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    if not app.testing:
        logging.basicConfig(
            filename=BASE_DIR / "app.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
        )

    origins = [o.strip() for o in str(app.config["CORS_ORIGINS"]).split(",") if o.strip()]
    CORS(app, origins=origins or "*")

    # Heavy resources are loaded on first use and cached (failures are not cached).
    cache = {}

    def get_analyzer():
        if "analyzer" not in cache:
            try:
                cache["analyzer"] = chatbot.create_analyzer()
            except ImportError:
                logging.exception("vaderSentiment is not installed")
                raise ApiError("Sentiment analysis is not available on the server.", 503)
        return cache["analyzer"]

    def get_pools():
        path = Path(app.config["RESPONSES_FILE"])
        mtime = path.stat().st_mtime_ns if path.exists() else None
        if cache.get("pools_key") != (str(path), mtime) or "pools" not in cache:
            try:
                cache["pools"] = chatbot.load_response_pools(path)
            except (OSError, ValueError):
                logging.exception("Could not load chatbot responses")
                raise ApiError("Chatbot responses are not available on the server.", 503)
            cache["pools_key"] = (str(path), mtime)
        return cache["pools"]

    def get_music_data():
        path = Path(app.config["MUSIC_FILE"])
        mtime = path.stat().st_mtime_ns if path.exists() else None
        if cache.get("music_key") != (str(path), mtime) or "music" not in cache:
            try:
                cache["music"] = music_therapy.load_music_data(path)
            except (OSError, ValueError):
                logging.exception("Could not load music data")
                raise ApiError("Music data is not set up on the server.", 503)
            cache["music_key"] = (str(path), mtime)
        return cache["music"]

    def get_clustered_habits():
        try:
            return habit_clustering.get_clustered_habits(app.config["HABITS_FILE"])
        except (OSError, ValueError):
            logging.exception("Could not load habits data")
            raise ApiError("Habit data is not available on the server.", 503)

    # ---------------- Error handling ----------------
    @app.errorhandler(ApiError)
    def handle_api_error(err):
        return jsonify({"error": err.message, **err.extra}), err.status

    @app.errorhandler(HTTPException)
    def handle_http_error(err):
        return jsonify({"error": err.description or err.name}), err.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(err):
        logging.exception("Unexpected error")
        return jsonify({"error": "Internal server error."}), 500

    # ---------------- Endpoints ----------------
    @app.route("/health", methods=["GET"])
    def health():
        """Simple status check that tells which parts are ready."""
        checks = {}
        for name, getter in (
            ("chatbot_responses", get_pools),
            ("habits", get_clustered_habits),
            ("music_data", get_music_data),
        ):
            try:
                getter()
                checks[name] = "ok"
            except ApiError:
                checks[name] = "unavailable"
        checks["music_folder"] = "ok" if Path(app.config["MUSIC_DIR"]).is_dir() else "unavailable"
        status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        return jsonify({"status": status, "checks": checks})

    @app.route("/options", methods=["GET"])
    def options():
        """Valid values for the dropdowns / inputs of the frontend."""
        try:
            music_moods = music_therapy.available_moods(get_music_data())
        except ApiError:
            music_moods = []
        return jsonify(
            {
                **habit_recommendation.OPTIONS,
                "writing_moods": list(ai_writing_therapist.VALID_MOODS),
                "music_moods": music_moods,
            }
        )

    @app.route("/chatbot", methods=["POST"])
    def chatbot_response():
        """Chatbot response based on user input sentiment."""
        user_input = require_text(get_json_object(), "input", MAX_CHATBOT_INPUT_CHARS)
        reply = chatbot.build_reply(user_input, get_analyzer(), get_pools())
        return jsonify(reply)

    @app.route("/ai_writing_therapist", methods=["POST"])
    def ai_writing_therapist_endpoint():
        """Provide writing prompts based on user mood."""
        mood = get_json_object().get("mood")
        try:
            prompt = ai_writing_therapist.get_prompt(mood)
        except ValueError as exc:
            raise ApiError(str(exc), 400)
        return jsonify({"prompt": prompt})

    @app.route("/recommend_habit", methods=["POST"])
    def recommend_habit_endpoint():
        """Recommend a habit with an explainable rule-based method."""
        user_data = get_json_object()
        missing = [k for k in habit_recommendation.FIELDS if k not in user_data]
        if missing:
            raise ApiError(
                f"Invalid input. Missing field(s): {', '.join(missing)}.",
                400,
                valid_options=habit_recommendation.OPTIONS,
            )
        try:
            result = habit_recommendation.recommend_habit_with_reason(
                *(user_data[k] for k in habit_recommendation.FIELDS)
            )
        except ValueError as exc:
            raise ApiError(str(exc), 400, valid_options=habit_recommendation.OPTIONS)
        logging.info("Recommended habit rule: %s", result["rule"])
        return jsonify(
            {
                "recommended_habit": result["recommended_habit"],
                "reason": result["reason"],
                "method": result["method"],
            }
        )

    @app.route("/music_recommendation", methods=["POST"])
    def music_recommendation():
        """Mood based playlist (shuffled). Audio is streamed from /music/<file_path>."""
        mood = require_text(get_json_object(), "mood", 50)
        data = get_music_data()
        songs, missing = music_therapy.list_songs(mood, data, app.config["MUSIC_DIR"])

        if not songs:
            if missing:
                raise ApiError(
                    "Songs exist for that mood, but their audio files are missing on the server.",
                    404,
                )
            raise ApiError(
                "No songs found for that mood.",
                404,
                available_moods=music_therapy.available_moods(data),
            )

        for song in songs:
            song["stream_url"] = f"/music/{song['file_path']}"
        return jsonify({"recommendations": songs})

    @app.route("/music/<path:filename>", methods=["GET"])
    def serve_music(filename):
        """Stream an audio file from the music folder (path traversal is blocked)."""
        if not music_therapy.is_allowed_audio(filename):
            raise ApiError("Unsupported file type.", 404)
        return send_from_directory(app.config["MUSIC_DIR"], filename)

    @app.route("/habit_clustering", methods=["POST"])
    def habit_clustering_endpoint():
        """Recommend habits that fit the user's available time."""
        user_data = get_json_object()
        if "time_available" not in user_data:
            raise ApiError("'time_available' (minutes) is required.")
        try:
            time_available = habit_clustering.parse_available_time(user_data["time_available"])
        except ValueError as exc:
            raise ApiError(str(exc), 400)

        clustered = get_clustered_habits()
        recommendations = habit_clustering.recommend_habits(clustered, time_available)

        if not recommendations:
            shortest = habit_clustering.shortest_habit_time(clustered)
            raise ApiError(
                f"No habits fit within {time_available} minutes "
                f"(the shortest habit needs {shortest} minutes).",
                404,
            )
        return jsonify({"recommendations": recommendations})
    
    # ---------------- Frontend ----------------
    @app.route("/", methods=["GET"])
    def index():
        return send_from_directory(
            BASE_DIR / "frontend",
            "index.html"
        )

    @app.route("/<any(style.css, app.js):filename>", methods=["GET"])
    def frontend_asset(filename):
        return send_from_directory(
            BASE_DIR / "frontend",
            filename
        )

    return app


app = create_app()

# Start the app
if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
