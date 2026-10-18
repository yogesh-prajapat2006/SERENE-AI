# habit_clustering.py
"""Habit clustering by time needed.

Habits are grouped with K-Means (on `time_needed`, in minutes) into up to
three groups (Quick / Medium / Long). Given the time a user has available,
`recommend_habits` picks habits from every group that fit in that time.
"""
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HABITS_FILE = BASE_DIR / "habits_data.csv"

REQUIRED_COLUMNS = ("habit_name", "time_needed")
MAX_CLUSTERS = 3

# Names for clusters, ordered from shortest to longest average time.
CLUSTER_LABELS = {
    1: ["All habits"],
    2: ["Quick", "Long"],
    3: ["Quick", "Medium", "Long"],
}

logger = logging.getLogger(__name__)


def parse_available_time(value):
    """Validate the user's available time and return it as a positive int (minutes).

    Accepts ints, whole floats and numeric strings such as "30".
    Raises ValueError with a user-friendly message otherwise.
    """
    if value is None or isinstance(value, bool):
        raise ValueError("time_available must be a number of minutes.")
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise ValueError("time_available must be a number of minutes.") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("time_available must be a finite number of minutes.")
    if number <= 0:
        raise ValueError("time_available must be greater than 0 minutes.")
    return int(number)  # whole minutes (30.9 -> 30)


def load_data(path=DEFAULT_HABITS_FILE):
    """Load habits data from CSV (needs columns `habit_name` and `time_needed`)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Habits file not found: {path}")
    data = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Habits file is missing column(s): {', '.join(missing)}")
    return data


def preprocess_and_cluster(data, n_clusters=MAX_CLUSTERS):
    """Cluster habits by `time_needed` and add `cluster` and `ordered_cluster` columns.

    `ordered_cluster` 0 is the quickest group. Edge cases handled:
    - rows with a missing / non-numeric / non-positive time are dropped (logged),
    - fewer rows or fewer distinct time values than `n_clusters` -> fewer clusters,
    - no valid rows -> ValueError (instead of a confusing KMeans error).
    Returns a new DataFrame; the input is not modified.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Habits data is missing column(s): {', '.join(missing)}")

    data = data.copy()
    data["time_needed"] = pd.to_numeric(data["time_needed"], errors="coerce")
    valid = data["time_needed"].notna() & (data["time_needed"] > 0)
    dropped = int((~valid).sum())
    if dropped:
        logger.warning("Dropped %d habit row(s) with invalid time_needed", dropped)
    data = data[valid].reset_index(drop=True)
    if data.empty:
        raise ValueError("No habits with a valid time_needed to cluster.")

    # K-Means cannot have more clusters than distinct points.
    k = max(1, min(int(n_clusters), data["time_needed"].nunique()))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    data["cluster"] = kmeans.fit_predict(data[["time_needed"]])

    # Order clusters from shortest to longest average time.
    cluster_order = data.groupby("cluster")["time_needed"].mean().sort_values().index
    cluster_mapping = {cluster: i for i, cluster in enumerate(cluster_order)}
    data["ordered_cluster"] = data["cluster"].map(cluster_mapping)
    return data


def cluster_label(ordered_cluster, total_clusters):
    """Human-friendly name for a cluster (Quick / Medium / Long)."""
    labels = CLUSTER_LABELS.get(total_clusters)
    if labels and 0 <= ordered_cluster < len(labels):
        return labels[ordered_cluster]
    return f"Group {ordered_cluster + 1}"


def recommend_habits(data, available_time):
    """Recommend habits that fit into `available_time` minutes, per cluster.

    Inside each cluster habits are tried from shortest to longest. A habit that
    does not fit is skipped (`continue`), it does NOT stop the search.
    Returns a list of plain-Python dicts: cluster, label, habits, total_time.
    """
    available_time = parse_available_time(available_time)
    total_clusters = int(data["ordered_cluster"].nunique())
    recommendations = []

    for cluster in sorted(data["ordered_cluster"].unique()):
        cluster_data = data[data["ordered_cluster"] == cluster].sort_values(
            ["time_needed", "habit_name"]
        )

        total_time = 0
        selected_habits = []
        for _, row in cluster_data.iterrows():
            needed = int(row["time_needed"])
            if total_time + needed > available_time:
                continue
            selected_habits.append(str(row["habit_name"]))
            total_time += needed

        if selected_habits:
            recommendations.append(
                {
                    "cluster": int(cluster),
                    "label": cluster_label(int(cluster), total_clusters),
                    "habits": selected_habits,
                    "total_time": int(total_time),
                }
            )

    return recommendations


def shortest_habit_time(data):
    """Minutes needed by the quickest habit (used for helpful error messages)."""
    return int(data["time_needed"].min())


@lru_cache(maxsize=8)
def _cached_clustering(path_str, mtime_ns):
    return preprocess_and_cluster(load_data(path_str))


def get_clustered_habits(path=DEFAULT_HABITS_FILE):
    """Cluster once and reuse the result until the CSV file changes."""
    path = Path(path)
    mtime_ns = path.stat().st_mtime_ns if path.exists() else 0
    return _cached_clustering(str(path), mtime_ns)


def main():
    try:
        data = preprocess_and_cluster(load_data())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    try:
        available_time = parse_available_time(
            input("Enter the time you have available (in minutes): ")
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    recommendations = recommend_habits(data, available_time)

    if recommendations:
        print("\nRecommended habits based on your available time:")
        for rec in recommendations:
            print(f"{rec['label']} (cluster {rec['cluster']}, total time: {rec['total_time']} mins):")
            for habit in rec["habits"]:
                print(f"- {habit}")
    else:
        print(
            f"No habits fit within {available_time} minutes "
            f"(the shortest habit needs {shortest_habit_time(data)} minutes)."
        )


if __name__ == "__main__":
    main()
