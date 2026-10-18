# SERENE-AI: AI-Assisted Wellness API

SERENE-AI is a small Flask REST API for a wellness app (Android, web, ...). It offers a
sentiment-aware chatbot, mood-based writing prompts, habit suggestions, time-based habit
grouping and a mood-based music playlist.

> **Important:** this is a learning / demo project. It gives general wellness suggestions,
> **not medical advice**, and it is **not** a crisis service. If you or someone else may be in
> danger, contact your local emergency number or a qualified professional.

## What is real "AI" here and what is not

| Feature | How it really works |
| --- | --- |
| Chatbot | [VADER](https://github.com/cjhutto/vaderSentiment) scores the tone of the message (positive / neutral / negative) and a canned reply is picked from `responses.csv`. A short list of English crisis phrases returns a static supportive message instead. VADER is **not** a crisis detector and can miss things. |
| Habit recommendation | **Explainable rules** (see `RULES` in `habit_recommendation.py`). Every answer includes a `reason`. The earlier Decision Tree was removed: with 16 training rows and 16 different labels it could only memorise, so its accuracy could not be measured. A model should come back only when real usage data exists. |
| Habit clustering | K-Means on `time_needed` (minutes) groups habits into Quick / Medium / Long. It is a small demo (18 habits), simple fixed time buckets would work too. |
| Music | A mood playlist: all songs of that mood in random order. There is no similarity model (the old cosine-similarity code gave every song the same score). |
| Writing therapist | Random prompt for the mood (good / neutral / bad). |

## Project layout

```
app.py                     Flask API (create_app factory)
chatbot.py                 sentiment chatbot logic (+ command-line chat)
ai_writing_therapist.py    writing prompts (+ command-line version)
habit_recommendation.py    rule-based habit recommender (+ command-line questionnaire)
habit_clustering.py        K-Means habit grouping (+ command-line version)
music_therapy.py           mood playlist logic (+ command-line player)
responses.csv              chatbot replies (responses.csv.bak = untouched original)
habits_data.csv            habit_name, time_needed (minutes)
music_data.csv             title, file_path, mood   (you provide this, see below)
music/                     your audio files (ignored by git)
tests/                     automated tests
```

## Setup (VS Code, Windows / macOS / Linux)

1. Install Python 3.12 (recommended) and open this folder in VS Code (`File > Open Folder`).
2. Open the terminal (`Ctrl + ~`) and create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate it:

   ```bash
   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   # Windows (cmd)
   venv\Scripts\activate.bat
   # macOS / Linux
   source venv/bin/activate
   ```

   If PowerShell blocks the script, run once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
   In VS Code choose the interpreter with `Ctrl+Shift+P` -> "Python: Select Interpreter" -> `venv`.

4. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the server:

   ```bash
   python app.py
   ```

   It runs on <http://127.0.0.1:5000>. Check <http://127.0.0.1:5000/health> in your browser.

Optional settings (environment variables): `PORT`, `HOST`, `FLASK_DEBUG=1` (development only),
`CORS_ORIGINS` (comma separated frontend URLs; default `*` = everyone, restrict it in production).

### Data files

- `habits_data.csv` is included.
- **Music:** create `music_data.csv` (see `music_data.sample.csv`) with columns
  `title,file_path,mood` and put the audio files (`.mp3`, `.wav`, `.ogg`) in the `music/` folder.
  `file_path` only needs the **file name**; old Windows paths such as `D:\ML_files\song.mp3`
  still work because only `song.mp3` is used. Without `music_data.csv` only the music endpoint
  answers `503`; the rest of the API keeps working.
- Do not commit audio files (large, and you need the right to share them). They are git-ignored.
- `responses.csv`: text that contains a comma must be inside double quotes, e.g.
  `Positive,"That's awesome, keep it up!"`. A badly formatted row is reported as an error.

## API

All endpoints return JSON. Errors always look like `{"error": "message"}`.
POST bodies must be JSON with `Content-Type: application/json`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | which parts of the server are ready |
| GET | `/options` | valid values for dropdowns (habit answers, moods) |
| POST | `/chatbot` | sentiment-aware reply |
| POST | `/ai_writing_therapist` | writing prompt for a mood |
| POST | `/recommend_habit` | rule-based habit suggestion |
| POST | `/habit_clustering` | habits that fit the available minutes |
| POST | `/music_recommendation` | shuffled playlist for a mood |
| GET | `/music/<file>` | stream an audio file |

### `POST /chatbot`

```json
{"input": "I feel great today!"}
```
```json
{"response": "That's wonderful! I'm glad to hear that!", "sentiment": "positive", "crisis": false}
```
`input` is required (max 1000 characters). `sentiment` is `positive`, `neutral`, `negative` or `crisis`.

### `POST /ai_writing_therapist`

```json
{"mood": "good"}
```
```json
{"prompt": "Describe a recent accomplishment you're proud of."}
```
`mood` is `good`, `neutral` or `bad` (case does not matter).

### `POST /recommend_habit`

All four fields are required. Valid values (case does not matter) are returned by `GET /options`:

- `exercise_frequency`: Never, Rarely, A few times a week, Daily
- `social_media_hours`: Less than 1 hour, 1-2 hours, 2-3 hours, More than 3 hours
- `stress_level`: Low, Moderate, High
- `mindfulness_frequency`: Never, Rarely, A few times a week, Daily

```json
{
  "exercise_frequency": "Daily",
  "social_media_hours": "1-2 hours",
  "stress_level": "High",
  "mindfulness_frequency": "Never"
}
```
```json
{
  "recommended_habit": "Practice 10 minutes of mindful breathing or meditation every day.",
  "reason": "Your stress is high and you rarely or never practise mindfulness, so a short daily breathing or meditation routine is a gentle place to start.",
  "method": "rule_based"
}
```
An invalid value returns `400` with a `valid_options` object. (`"Weekly"` is accepted as a shortcut for "A few times a week".)

### `POST /habit_clustering`

```json
{"time_available": 30}
```
```json
{
  "recommendations": [
    {"cluster": 0, "label": "Quick", "habits": ["Plan Tomorrow's Tasks", "Mindfulness Meditation"], "total_time": 25}
  ]
}
```
`time_available` is minutes (a number or a numeric string like `"30"`, greater than 0). `400` for invalid values,
`404` when no habit fits (the message tells the shortest habit time).

### `POST /music_recommendation`

```json
{"mood": "relaxation"}
```
```json
{
  "recommendations": [
    {"title": "Calm Waves", "file_path": "calm.mp3", "stream_url": "/music/calm.mp3"}
  ]
}
```
Songs come in random order. Prefix `stream_url` with your server address in the app
(e.g. `http://192.168.1.5:5000/music/calm.mp3`). Songs whose audio file is missing from `music/` are left out.
An unknown mood returns `404` with `available_moods`.

## Tests

```bash
python -m unittest discover -s tests -v
```
(`python -m pytest -v` also works if you install pytest.)

The tests cover the repaired `responses.csv`, the clustering bugs (time budgets, string input, K-Means with few rows),
the rule-based recommender (all 192 input combinations), every API endpoint including error cases, and
path-traversal protection for audio streaming. If `flask-cors` or `vaderSentiment` are not installed, the test run
prints which tiny stand-ins it used. With `pip install -r requirements.txt` the real packages are used.

## Production notes

- Linux: `gunicorn -w 2 -b 0.0.0.0:8000 app:app` (never use `FLASK_DEBUG=1` on a server).
- Set `CORS_ORIGINS` to your frontend address(es) and put the app behind HTTPS (e.g. Nginx).
- The original project was hosted on Vultr; any Linux VPS works the same way.

## Known limitations

- Chatbot replies are canned; crisis detection is a short English phrase list only.
- Habit rules are general suggestions written by the project authors and should be reviewed by a
  qualified person before real-world use.
- Music playlist uses mood labels only. Better matching would need hand-labelled audio data
  (energy, tempo, ...) or real listening feedback.
- A machine-learning habit model needs real usage data (inputs, habits suggested, "was this helpful?") first.
