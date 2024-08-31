import tarfile
from pathlib import Path

import pandas as pd

import requests

__all__ = ["get_train_test_data", "get_unsup_data"]


def data_dir():
    """Return the path to the data directory.

    Downloads the data if it's not there.
    """
    data_path = Path("~/.nlpbook/dataset/aclImdb/aclImdb").expanduser()
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Untar the data, download it if it's not there.
        tar_path = data_path.parent / "aclImdb_v1.tar.gz"
        if not tar_path.exists():
            r = requests.get(
                "https://github.com/spenceforce/NLP-Simple-to-Spectacular/releases/download/aclImdb_dataset/aclImdb_v1.tar.gz",
                stream=True,
            )
            with tar_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)

        with tarfile.open(tar_path) as tar:
            tar.extractall(filter="data")

        # Ensure the movie ID mapping is present, download it if it's not there.
        movie_map_path = data_path.parent / "all_movie_ids.csv"
        if not movie_map_path.exists():
            r = requests.get(
                "https://github.com/spenceforce/NLP-Simple-to-Spectacular/releases/download/aclImdb_dataset/all_movie_ids.csv",
                stream=True,
            )
            with movie_map_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)

    return data_path


def get_df(dataset, dedup=False):
    """Return a `pd.DataFrame` for a dataset."""
    dfs = []
    data_path = data_dir()
    for label, label_name in enumerate(["neg", "pos"]):
        df = get_review_data(
            data_path / dataset / label_name,
            data_path / dataset / f"urls_{label_name}.txt",
            dedup,
        )
        df["label"] = label
        dfs.append(df)
    return pd.concat(dfs)


def get_train_data(dedup=True):
    """
    Return a `pd.DataFrame` with the supervised training data.
    """
    return get_df("train", dedup)


def get_test_data(dedup=True):
    """
    Return a `pd.DataFrame` with the supervised testing data.
    """
    return get_df("test", dedup)


def get_review_data(reviews_dir, urls_file, dedup=False):
    """
    Return a `pd.DataFrame` containing the review ID,
    movie ID, rating, and review.
    """
    with urls_file.open() as f:
        movie_ids = {i: url.split("/")[4] for i, url in enumerate(f.readlines())}

    data_path = data_dir
    movie_id_map = dict(pd.read_csv(data_dir.parent / "all_movie_ids.csv").values)

    data = []
    for p in (reviews_dir).iterdir():
        ID, rating = map(int, p.stem.split("_"))
        data.append(
            {
                "id": ID,
                "movie_id": movie_id_map[movie_ids[ID]],
                "rating": rating,
                "review": p.open().read().strip(),
            }
        )

    rv = pd.DataFrame(data)
    if dedup:
        return rv.drop_duplicates("review").copy()
    return rv


def get_train_test_data():
    """Return train and test `pd.DataFrame`s."""
    train_df = get_train_data()
    test_df = get_test_data()
    same_review = test_df["review"].isin(train_df["review"])
    same_movie = test_df["movie_id"].isin(train_df["movie_id"])
    test_df = test_df[~same_review & ~same_movie].copy()
    return train_df, test_df


def get_unsup_data(dedup=True):
    """
    Return a `pd.DataFrame` with the unsupervised data.
    """
    data_path = data_dir()
    rv = get_review_data(
        data_path / "train/unsup", data_path / "train/urls_unsup.txt", dedup
    )
    rv.drop(columns="rating", inplace=True)
    # Drop the ratings column since every review in the
    # unsupervised set is given a rating of 0 regardless
    # of what it's rating is.
    return rv
