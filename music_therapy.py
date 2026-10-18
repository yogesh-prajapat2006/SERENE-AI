# music_therapy.py
"""Mood-based music playlist.

music_data.csv needs the columns: title, file_path, mood.
`file_path` may be a plain file name (song.mp3) or an old Windows path
(D:\\ML_files\\song.mp3): only the file name is used, and the audio file is
looked up inside the project's `music/` folder.

This is a simple mood playlist (all songs of the mood, shuffled), not a
machine-learning recommender. The old cosine-similarity code gave every song
the same score inside one mood, so it was removed.
"""
import logging
import ntpath
import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MUSIC_FILE = BASE_DIR / "music_data.csv"
DEFAULT_MUSIC_DIR = BASE_DIR / "music"

REQUIRED_COLUMNS = ("title", "file_path", "mood")
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg"}

logger = logging.getLogger(__name__)


def load_music_data(path=DEFAULT_MUSIC_FILE):
    """Load the music CSV; raises FileNotFoundError / ValueError with clear messages."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Music data file not found: {path}")
    data = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Music data is missing column(s): {', '.join(missing)}")
    data = data.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    data["mood"] = data["mood"].astype(str).str.strip()
    return data


def clean_filename(file_path):
    """'D:\\ML_files\\a.mp3' or 'music/a.mp3' -> 'a.mp3' (works for both slash types)."""
    return ntpath.basename(str(file_path).strip())


def is_allowed_audio(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def available_moods(data):
    return sorted({m.lower() for m in data["mood"]})


def get_recommendations(mood, data=None, rng=None):
    """Return all songs for `mood` in random order (a DataFrame), or None if there are none."""
    if data is None:
        data = load_music_data()
    mood = str(mood).strip().lower()
    filtered = data[data["mood"].str.lower() == mood]
    if filtered.empty:
        return None
    rng = rng or random
    indices = list(filtered.index)
    rng.shuffle(indices)
    return filtered.loc[indices]


def list_songs(mood, data, music_dir=DEFAULT_MUSIC_DIR, rng=None):
    """Songs for the API: list of {"title", "file_path"(file name only)}.

    Songs whose audio file is missing from `music_dir` are skipped (and logged).
    Returns (songs, missing_count); songs is [] when the mood has no songs.
    """
    music_dir = Path(music_dir)
    rows = get_recommendations(mood, data, rng)
    if rows is None:
        return [], 0
    songs, missing = [], 0
    for _, row in rows.iterrows():
        filename = clean_filename(row["file_path"])
        if not filename or not is_allowed_audio(filename):
            logger.warning("Skipping song with unsupported file: %r", row["file_path"])
            missing += 1
            continue
        if not (music_dir / filename).is_file():
            logger.warning("Audio file not found in %s: %s", music_dir, filename)
            missing += 1
            continue
        songs.append({"title": str(row["title"]), "file_path": filename})
    return songs, missing


# ---- Command-line player (needs the optional `pygame` package) ----
def _init_mixer():
    try:
        import pygame
    except ImportError:
        print("pygame is not installed. Run: pip install -r requirements-desktop.txt")
        return None
    pygame.mixer.init()
    return pygame


def play_song(file_path, music_dir=DEFAULT_MUSIC_DIR):
    """Play a song with an option to stop. Returns True if it was played."""
    pygame = _init_mixer()
    if pygame is None:
        return False
    full_path = Path(music_dir) / clean_filename(file_path)
    if not full_path.is_file():
        print(f"Audio file not found: {full_path}")
        return False
    pygame.mixer.music.load(str(full_path))
    pygame.mixer.music.play()
    input("Press Enter to stop the music.\n")
    pygame.mixer.music.stop()
    return True


def main():
    try:
        data = load_music_data()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    print("Welcome to the Music Therapy!")
    while True:
        mood = input(f"Enter your mood ({'/'.join(available_moods(data))}): ").strip().lower()
        recommendations = get_recommendations(mood, data)

        if recommendations is not None:
            print("Here are some recommendations:")
            for _, row in recommendations.iterrows():
                print(f"- {row['title']}")
            first = recommendations.iloc[0]
            print(f"\nPlaying: {first['title']}")
            if play_song(first["file_path"]):
                print("Music stopped.")
        else:
            print("No songs found for that mood.")

        if input("Try another mood? (yes/no): ").strip().lower() != "yes":
            break


if __name__ == "__main__":
    main()
