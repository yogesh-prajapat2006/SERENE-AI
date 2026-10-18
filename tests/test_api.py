"""Tests for SERENE-AI.

Run from the project folder (works without pytest):
    python -m unittest discover -s tests -v
or, if pytest is installed:
    python -m pytest -v
"""
import csv
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _stubs  # noqa: E402

_stubs.install_stubs()

import pandas as pd  # noqa: E402

import ai_writing_therapist  # noqa: E402
import app as app_module  # noqa: E402
import chatbot  # noqa: E402
import habit_clustering as hc  # noqa: E402
import habit_recommendation as hr  # noqa: E402
import music_therapy  # noqa: E402


def setUpModule():
    if _stubs.STUBBED:
        print(f"\n[tests] NOTE: real package(s) not installed, using stubs for: {', '.join(_stubs.STUBBED)}")
    else:
        print("\n[tests] Using the real flask-cors and vaderSentiment packages.")


# ---------------------------------------------------------------- responses.csv
class ResponsesCsvTests(unittest.TestCase):
    def test_backup_exists(self):
        self.assertTrue((PROJECT_ROOT / "responses.csv.bak").exists())

    def test_every_original_response_is_preserved_in_order(self):
        """Split the backup on the FIRST comma only and compare with the fixed file."""
        original = []
        lines = (PROJECT_ROOT / "responses.csv.bak").read_text(encoding="utf-8").split("\n")
        for line in lines[1:]:
            if line.strip():
                sentiment, response = line.split(",", 1)
                original.append((sentiment, response))
        with open(PROJECT_ROOT / "responses.csv", encoding="utf-8", newline="") as f:
            fixed = [(r["Sentiment"], r["Response"]) for r in csv.DictReader(f)]
        self.assertEqual(len(original), 45)
        self.assertEqual(fixed, original)

    def test_every_row_has_exactly_two_fields(self):
        with open(PROJECT_ROOT / "responses.csv", encoding="utf-8", newline="") as f:
            for number, row in enumerate(csv.reader(f), start=1):
                self.assertEqual(len(row), 2, f"line {number}: {row}")

    def test_pools_load_all_rows(self):
        pools = chatbot.load_response_pools(PROJECT_ROOT / "responses.csv")
        self.assertEqual({k: len(v) for k, v in pools.items()}, {"positive": 15, "neutral": 15, "negative": 15})
        self.assertIn("That's awesome, keep it up!", pools["positive"])

    def test_unquoted_comma_row_is_reported_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("Sentiment,Response\nPositive,Great, keep going\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                chatbot.load_response_pools(path)

    def test_missing_sentiment_pool_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.csv"
            path.write_text("Sentiment,Response\nPositive,Nice\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                chatbot.load_response_pools(path)


# ---------------------------------------------------------------- clustering
class HabitClusteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = hc.preprocess_and_cluster(hc.load_data(PROJECT_ROOT / "habits_data.csv"))

    def test_small_budgets_return_habits_break_bug_fixed(self):
        # Old code returned nothing for these because the first CSV habit (30 min) did not fit.
        for budget in (10, 15, 20, 25, 29):
            with self.subTest(budget=budget):
                self.assertTrue(hc.recommend_habits(self.data, budget))

    def test_long_habit_listed_first_does_not_block_shorter_ones(self):
        """The exact old bug: first row (30 min) does not fit, so the old loop stopped."""
        df = pd.DataFrame(
            {
                "habit_name": ["Morning Run", "Plan Tasks", "Meditation"],
                "time_needed": [30, 10, 15],
                "ordered_cluster": [0, 0, 0],
            }
        )
        recs = hc.recommend_habits(df, 15)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["habits"], ["Plan Tasks"])
        self.assertEqual(hc.recommend_habits(df, 25)[0]["habits"], ["Plan Tasks", "Meditation"])

    def test_recommend_habits_validates_time_itself(self):
        self.assertTrue(hc.recommend_habits(self.data, "30"))
        for bad in ("abc", -1, 0, None):
            with self.assertRaises(ValueError):
                hc.recommend_habits(self.data, bad)

    def test_selection_never_exceeds_budget_and_total_matches(self):
        times = dict(zip(self.data["habit_name"], self.data["time_needed"]))
        for budget in (10, 30, 45, 60, 120, 500):
            for rec in hc.recommend_habits(self.data, budget):
                self.assertLessEqual(rec["total_time"], budget)
                self.assertEqual(rec["total_time"], sum(int(times[h]) for h in rec["habits"]))

    def test_budget_below_shortest_habit_returns_empty(self):
        self.assertEqual(hc.recommend_habits(self.data, 5), [])
        self.assertEqual(hc.shortest_habit_time(self.data), 10)

    def test_output_is_plain_json_serialisable(self):
        json.dumps(hc.recommend_habits(self.data, 120))

    def test_clusters_are_ordered_and_named(self):
        recs = hc.recommend_habits(self.data, 500)
        self.assertEqual([r["label"] for r in recs], ["Quick", "Medium", "Long"])

    def test_parse_available_time_accepts_valid(self):
        self.assertEqual(hc.parse_available_time(30), 30)
        self.assertEqual(hc.parse_available_time("30"), 30)
        self.assertEqual(hc.parse_available_time(" 45 "), 45)
        self.assertEqual(hc.parse_available_time(30.9), 30)

    def test_parse_available_time_rejects_invalid(self):
        for bad in (None, True, False, "abc", "", -5, 0, "0", float("nan"), float("inf"), [30], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    hc.parse_available_time(bad)

    def test_kmeans_with_two_rows_does_not_crash(self):
        df = pd.DataFrame({"habit_name": ["a", "b"], "time_needed": [10, 20]})
        self.assertEqual(hc.preprocess_and_cluster(df)["ordered_cluster"].tolist(), [0, 1])

    def test_kmeans_with_one_row_or_identical_times(self):
        one = pd.DataFrame({"habit_name": ["a"], "time_needed": [10]})
        same = pd.DataFrame({"habit_name": list("abc"), "time_needed": [10, 10, 10]})
        self.assertEqual(hc.preprocess_and_cluster(one)["ordered_cluster"].tolist(), [0])
        self.assertEqual(hc.preprocess_and_cluster(same)["ordered_cluster"].tolist(), [0, 0, 0])

    def test_empty_or_invalid_data_gives_clear_error(self):
        with self.assertRaises(ValueError):
            hc.preprocess_and_cluster(pd.DataFrame({"habit_name": [], "time_needed": []}))
        with self.assertRaises(ValueError):
            hc.preprocess_and_cluster(pd.DataFrame({"habit_name": ["a"], "minutes": [5]}))
        with self.assertRaises(ValueError):
            hc.preprocess_and_cluster(pd.DataFrame({"habit_name": ["a"], "time_needed": ["x"]}))

    def test_rows_with_bad_times_are_dropped_not_fatal(self):
        df = pd.DataFrame({"habit_name": ["a", "b", "c"], "time_needed": [10, "oops", -3]})
        result = hc.preprocess_and_cluster(df)
        self.assertEqual(result["habit_name"].tolist(), ["a"])

    def test_input_dataframe_is_not_modified(self):
        df = pd.DataFrame({"habit_name": ["a", "b"], "time_needed": [10, 20]})
        hc.preprocess_and_cluster(df)
        self.assertEqual(list(df.columns), ["habit_name", "time_needed"])


# ---------------------------------------------------------------- recommender
class HabitRecommendationTests(unittest.TestCase):
    def rule_for(self, exercise, social, stress, mindfulness):
        return hr.recommend_habit_with_reason(exercise, social, stress, mindfulness)["rule"]

    def test_each_rule_can_be_reached(self):
        cases = {
            "high_stress_low_mindfulness": ("Daily", "Less than 1 hour", "High", "Never"),
            "high_stress_low_exercise": ("Never", "Less than 1 hour", "High", "Daily"),
            "heavy_social_media": ("Daily", "More than 3 hours", "Low", "Daily"),
            "low_exercise": ("Rarely", "Less than 1 hour", "Low", "Daily"),
            "low_mindfulness": ("Daily", "Less than 1 hour", "Low", "Rarely"),
            "moderate_stress": ("Daily", "Less than 1 hour", "Moderate", "Daily"),
            "social_media_medium": ("Daily", "2-3 hours", "Low", "Daily"),
            "default": ("Daily", "Less than 1 hour", "Low", "Daily"),
        }
        for expected, answers in cases.items():
            with self.subTest(rule=expected):
                self.assertEqual(self.rule_for(*answers), expected)

    def test_result_is_explainable(self):
        result = hr.recommend_habit_with_reason("Never", "1-2 hours", "High", "Never")
        self.assertEqual(result["method"], "rule_based")
        self.assertTrue(result["reason"])
        self.assertTrue(result["recommended_habit"])

    def test_case_and_spacing_are_ignored_and_weekly_alias_works(self):
        a = hr.recommend_habit("daily", "less than 1 HOUR", " low ", "WEEKLY")
        b = hr.recommend_habit("Daily", "Less than 1 hour", "Low", "A few times a week")
        self.assertEqual(a, b)

    def test_invalid_values_raise_with_valid_options_in_message(self):
        # The old README example values "1" and "Weekly" for the wrong fields are rejected clearly.
        with self.assertRaises(ValueError) as ctx:
            hr.recommend_habit("Daily", "1", "High", "Weekly")
        self.assertIn("Less than 1 hour", str(ctx.exception))
        for bad in (None, 5, ["Daily"]):
            with self.assertRaises(ValueError):
                hr.recommend_habit(bad, "1-2 hours", "Low", "Daily")

    def test_same_input_gives_same_output(self):
        args = ("Rarely", "2-3 hours", "Moderate", "Never")
        self.assertEqual(hr.recommend_habit(*args), hr.recommend_habit(*args))

    def test_all_192_input_combinations_get_a_recommendation(self):
        combos = list(itertools.product(*hr.OPTIONS.values()))
        self.assertEqual(len(combos), 192)
        for combo in combos:
            result = hr.recommend_habit_with_reason(*combo)
            self.assertTrue(result["recommended_habit"] and result["reason"], combo)


# ---------------------------------------------------------------- API
MUSIC_CSV = """title,file_path,mood,feature1,feature2,feature3
Calm Waves,D:\\ML_files\\calm.mp3,relaxation,0.1,0.2,0.3
Soft Rain,D:\\ML_files\\rain.mp3,Relaxation,0.2,0.1,0.4
Pump It,D:\\ML_files\\pump.mp3,motivation,0.9,0.8,0.7
Happy Day,music/happy.mp3,fun,0.5,0.5,0.5
Party Time,party.mp3,fun,0.6,0.4,0.5
Ghost Song,D:\\ML_files\\ghost.mp3,fun,0.1,0.1,0.1
"""


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.music_dir = self.tmp / "music"
        self.music_dir.mkdir()
        for name in ("calm.mp3", "rain.mp3", "pump.mp3", "happy.mp3", "party.mp3"):  # no ghost.mp3
            (self.music_dir / name).write_bytes(b"ID3-fake-audio-" + name.encode())
        (self.tmp / "music_data.csv").write_text(MUSIC_CSV, encoding="utf-8")
        (self.tmp / "secret.txt").write_text("top secret", encoding="utf-8")
        self.app = self.make_app()
        self.client = self.app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def make_app(self, **overrides):
        config = {
            "TESTING": True,
            "RESPONSES_FILE": PROJECT_ROOT / "responses.csv",
            "HABITS_FILE": PROJECT_ROOT / "habits_data.csv",
            "MUSIC_FILE": self.tmp / "music_data.csv",
            "MUSIC_DIR": self.music_dir,
        }
        config.update(overrides)
        return app_module.create_app(config)

    def post(self, path, payload=None, **kwargs):
        if payload is None:
            return self.client.post(path, **kwargs)
        return self.client.post(path, json=payload, **kwargs)


class GeneralApiTests(ApiTestCase):
    def test_debug_is_off_by_default(self):
        self.assertFalse(self.app.debug)

    def test_unknown_route_returns_json_404(self):
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())

    def test_wrong_method_returns_json_405(self):
        response = self.client.get("/chatbot")
        self.assertEqual(response.status_code, 405)
        self.assertIn("error", response.get_json())

    def test_unexpected_exception_returns_json_500_without_details(self):
        with mock.patch.object(chatbot, "build_reply", side_effect=RuntimeError("secret detail")):
            response = self.post("/chatbot", {"input": "hello"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "Internal server error."})

    def test_health_all_ok(self):
        body = self.client.get("/health").get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(set(body["checks"].values()), {"ok"})

    def test_health_degraded_when_music_missing(self):
        app = self.make_app(MUSIC_FILE=self.tmp / "missing.csv")
        body = app.test_client().get("/health").get_json()
        self.assertEqual(body["status"], "degraded")
        self.assertEqual(body["checks"]["music_data"], "unavailable")
        self.assertEqual(body["checks"]["habits"], "ok")

    def test_options_lists_valid_values(self):
        body = self.client.get("/options").get_json()
        self.assertEqual(body["stress_level"], ["Low", "Moderate", "High"])
        self.assertEqual(body["writing_moods"], ["good", "neutral", "bad"])
        self.assertEqual(body["music_moods"], ["fun", "motivation", "relaxation"])

    def test_options_still_works_without_music_data(self):
        app = self.make_app(MUSIC_FILE=self.tmp / "missing.csv")
        body = app.test_client().get("/options").get_json()
        self.assertEqual(body["music_moods"], [])


class ChatbotApiTests(ApiTestCase):
    def pools(self):
        return chatbot.load_response_pools(PROJECT_ROOT / "responses.csv")

    def test_positive_neutral_negative_replies_come_from_the_right_pool(self):
        pools = self.pools()
        cases = {
            "positive": "I feel great and happy today",
            "negative": "I feel sad and terrible",
            "neutral": "The table is brown",
        }
        for sentiment, text in cases.items():
            with self.subTest(sentiment=sentiment):
                response = self.post("/chatbot", {"input": text})
                body = response.get_json()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(body["sentiment"], sentiment)
                self.assertIn(body["response"], pools[sentiment])
                self.assertFalse(body["crisis"])

    def test_repaired_rows_are_reachable(self):
        # "That's awesome, keep it up!" used to be cut off at the comma.
        pools = self.pools()
        seen = {self.post("/chatbot", {"input": "so happy and great"}).get_json()["response"] for _ in range(200)}
        self.assertTrue(seen <= set(pools["positive"]))
        self.assertIn("That's awesome, keep it up!", seen)

    def test_crisis_phrase_gets_supportive_static_message(self):
        body = self.post("/chatbot", {"input": "I want to die"}).get_json()
        self.assertTrue(body["crisis"])
        self.assertEqual(body["response"], chatbot.CRISIS_RESPONSE)
        self.assertNotRegex(body["response"], r"\d{3,}")  # no made-up phone numbers

    def test_bad_requests(self):
        for kwargs in (
            {"data": "not json", "content_type": "text/plain"},
            {"json": ["a", "list"]},
            {"json": {}},
            {"json": {"input": ""}},
            {"json": {"input": "   "}},
            {"json": {"input": 123}},
            {"json": {"input": "x" * 1001}},
        ):
            with self.subTest(request=str(kwargs)[:40]):
                response = self.client.post("/chatbot", **kwargs)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_missing_responses_file_gives_503_json(self):
        app = self.make_app(RESPONSES_FILE=self.tmp / "nope.csv")
        response = app.test_client().post("/chatbot", json={"input": "hello"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("error", response.get_json())


class WritingTherapistApiTests(ApiTestCase):
    def test_valid_moods(self):
        for mood in ("good", "neutral", "bad", " GOOD "):
            with self.subTest(mood=mood):
                response = self.post("/ai_writing_therapist", {"mood": mood})
                self.assertEqual(response.status_code, 200)
                self.assertIn(response.get_json()["prompt"], ai_writing_therapist.topics[mood.strip().lower()])

    def test_invalid_input(self):
        for payload in ({"mood": "angry"}, {"mood": 5}, {"mood": None}, {}, ["good"]):
            with self.subTest(payload=payload):
                response = self.post("/ai_writing_therapist", payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_non_json_body(self):
        response = self.client.post("/ai_writing_therapist", data="mood=good")
        self.assertEqual(response.status_code, 400)


class RecommendHabitApiTests(ApiTestCase):
    VALID = {
        "exercise_frequency": "Daily",
        "social_media_hours": "1-2 hours",
        "stress_level": "High",
        "mindfulness_frequency": "A few times a week",
    }

    def test_valid_request(self):
        response = self.post("/recommend_habit", self.VALID)
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(body), {"recommended_habit", "reason", "method"})
        self.assertEqual(body["method"], "rule_based")

    def test_case_insensitive_values(self):
        lowered = {k: v.lower() for k, v in self.VALID.items()}
        self.assertEqual(self.post("/recommend_habit", lowered).status_code, 200)

    def test_missing_field(self):
        payload = dict(self.VALID)
        del payload["stress_level"]
        response = self.post("/recommend_habit", payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("stress_level", response.get_json()["error"])

    def test_invalid_value_returns_valid_options(self):
        # The old README example used "1" and "Weekly", which are not valid.
        payload = dict(self.VALID, social_media_hours="1")
        response = self.post("/recommend_habit", payload)
        body = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertIn("social_media_hours", body["valid_options"])

    def test_weekly_alias_is_accepted(self):
        self.assertEqual(self.post("/recommend_habit", dict(self.VALID, mindfulness_frequency="Weekly")).status_code, 200)

    def test_wrong_types_and_bodies(self):
        self.assertEqual(self.post("/recommend_habit", dict(self.VALID, stress_level=3)).status_code, 400)
        self.assertEqual(self.post("/recommend_habit", ["x"]).status_code, 400)
        self.assertEqual(self.client.post("/recommend_habit").status_code, 400)


class HabitClusteringApiTests(ApiTestCase):
    def test_valid_time(self):
        response = self.post("/habit_clustering", {"time_available": 30})
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        for rec in body["recommendations"]:
            self.assertLessEqual(rec["total_time"], 30)

    def test_string_time_is_accepted_instead_of_500(self):
        response = self.post("/habit_clustering", {"time_available": "30"})
        self.assertEqual(response.status_code, 200)

    def test_small_budget_returns_habits(self):
        for budget in (10, 15, 20, 25, 29):
            with self.subTest(budget=budget):
                self.assertEqual(self.post("/habit_clustering", {"time_available": budget}).status_code, 200)

    def test_budget_too_small_gives_accurate_404(self):
        response = self.post("/habit_clustering", {"time_available": 5})
        self.assertEqual(response.status_code, 404)
        self.assertIn("shortest habit needs 10 minutes", response.get_json()["error"])

    def test_invalid_values_give_400(self):
        for payload in ({}, {"time_available": "abc"}, {"time_available": -1}, {"time_available": 0},
                        {"time_available": None}, {"time_available": True}, {"time_available": [1]}):
            with self.subTest(payload=payload):
                response = self.post("/habit_clustering", payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_non_json_body(self):
        self.assertEqual(self.client.post("/habit_clustering", data="30").status_code, 400)

    def test_missing_habits_file_gives_503(self):
        app = self.make_app(HABITS_FILE=self.tmp / "nope.csv")
        response = app.test_client().post("/habit_clustering", json={"time_available": 30})
        self.assertEqual(response.status_code, 503)


class MusicApiTests(ApiTestCase):
    def test_returns_all_available_songs_for_mood_with_clean_paths(self):
        response = self.post("/music_recommendation", {"mood": "fun"})
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        songs = {s["title"]: s for s in body["recommendations"]}
        # Ghost Song is in the CSV but its audio file does not exist -> not returned.
        self.assertEqual(set(songs), {"Happy Day", "Party Time"})
        self.assertEqual(songs["Happy Day"]["file_path"], "happy.mp3")
        self.assertEqual(songs["Happy Day"]["stream_url"], "/music/happy.mp3")
        for song in songs.values():
            self.assertNotIn("\\", song["file_path"])
            self.assertNotIn(":", song["file_path"])

    def test_windows_paths_are_cleaned(self):
        body = self.post("/music_recommendation", {"mood": "relaxation"}).get_json()
        self.assertEqual({s["file_path"] for s in body["recommendations"]}, {"calm.mp3", "rain.mp3"})

    def test_mood_is_case_and_space_insensitive(self):
        self.assertEqual(self.post("/music_recommendation", {"mood": " RELAXATION "}).status_code, 200)

    def test_order_is_shuffled(self):
        orders = {tuple(s["title"] for s in self.post("/music_recommendation", {"mood": "relaxation"}).get_json()["recommendations"])
                  for _ in range(60)}
        self.assertEqual(len(orders), 2)  # both orders appear, so it is not a fixed order

    def test_unknown_mood_404_with_available_moods(self):
        response = self.post("/music_recommendation", {"mood": "angry"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["available_moods"], ["fun", "motivation", "relaxation"])

    def test_all_audio_files_missing_gives_clear_404(self):
        for f in self.music_dir.glob("*.mp3"):
            f.unlink()
        response = self.post("/music_recommendation", {"mood": "fun"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("audio files are missing", response.get_json()["error"])

    def test_invalid_requests(self):
        for payload in ({}, {"mood": ""}, {"mood": 7}, ["fun"]):
            with self.subTest(payload=payload):
                self.assertEqual(self.post("/music_recommendation", payload).status_code, 400)

    def test_missing_music_csv_gives_503_not_crash(self):
        app = self.make_app(MUSIC_FILE=self.tmp / "missing.csv")
        response = app.test_client().post("/music_recommendation", json={"mood": "fun"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(str(self.tmp), response.get_json()["error"])

    def test_csv_without_feature_columns_works(self):
        (self.tmp / "music_data.csv").write_text("title,file_path,mood\nA,a.mp3,calm\n", encoding="utf-8")
        (self.music_dir / "a.mp3").write_bytes(b"x")
        response = self.make_app().test_client().post("/music_recommendation", json={"mood": "calm"})
        self.assertEqual(response.status_code, 200)

    def test_csv_missing_required_column_gives_503(self):
        (self.tmp / "music_data.csv").write_text("title,mood\nA,calm\n", encoding="utf-8")
        response = self.make_app().test_client().post("/music_recommendation", json={"mood": "calm"})
        self.assertEqual(response.status_code, 503)

    def test_streaming_serves_the_file(self):
        response = self.client.get("/music/happy.mp3")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"ID3-fake-audio-happy.mp3")
        response.close()

    def test_streaming_blocks_path_traversal_and_other_files(self):
        for url in ("/music/../secret.txt", "/music/..%2Fsecret.txt", "/music/%2e%2e/secret.txt",
                    "/music/secret.txt", "/music/../music_data.csv", "/music/nonexistent.mp3"):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(b"top secret", response.data)
                response.close()


class MusicModuleTests(unittest.TestCase):
    def test_clean_filename_handles_both_slash_styles(self):
        self.assertEqual(music_therapy.clean_filename(r"D:\ML_files\song one.mp3"), "song one.mp3")
        self.assertEqual(music_therapy.clean_filename("music/song.mp3"), "song.mp3")
        self.assertEqual(music_therapy.clean_filename("  song.mp3 "), "song.mp3")

    def test_get_recommendations_none_for_unknown_mood(self):
        data = pd.DataFrame({"title": ["a"], "file_path": ["a.mp3"], "mood": ["fun"]})
        self.assertIsNone(music_therapy.get_recommendations("sad", data))
        self.assertEqual(len(music_therapy.get_recommendations("FUN", data)), 1)

    def test_importing_module_does_not_need_pygame(self):
        self.assertNotIn("pygame", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
