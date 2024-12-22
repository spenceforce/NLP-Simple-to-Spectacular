import pathlib
import tarfile

import pandas as pd

with tarfile.open("aclImdb_v1.tar.gz") as tar:
    tar.extractall(filter="data")
    data_path = pathlib.Path("aclImdb")
    # The untarred directory is `aclImdb` instead of `aclImdb_v1`.


def get_review_data(reviews_dir, urls_file):
    """Return a `pd.DataFrame` containing the review ID, title ID, rating, and review."""
    with urls_file.open() as f:
        title_ids = {
            i: url.split("/")[4]
            for i, url in enumerate(f.readlines())
        }

    data = []
    for p in (reviews_dir).iterdir():
        ID, rating = map(int, p.stem.split("_"))
        data.append(
            {
                "id": ID,
                "movie_id": title_ids[ID],
                "rating": rating,
                "review": p.open().read().strip(),
            }
        )

    return pd.DataFrame(data)


def get_df(dataset):
    dfs = []
    for label, label_name in enumerate(["neg", "pos"]):
        df = get_review_data(
            data_path / dataset / label_name,
            data_path / dataset / f"urls_{label_name}.txt",
        )
        df["label"] = label
        dfs.append(df)
    return pd.concat(dfs)


def get_train_data(dedup=True):
    return get_df("train").drop_duplicates("review").copy()


def get_test_data(dedup=True):
    return get_df("test").drop_duplicates("review").copy()


test_data = get_test_data()
unknown_ids = pd.read_csv("unknown_movie_ids.csv")

from pprint import pprint

pprint(
    test_data[test_data["movie_id"].isin(unknown_ids["movie_id"])][
        "review"
    ].tolist()
)
