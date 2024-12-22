import argparse
import concurrent.futures
import pathlib
import tarfile
import time

import pandas as pd
import requests
from tqdm import tqdm

with tarfile.open("aclImdb_v1.tar.gz") as tar:
    tar.extractall(filter="data")
    data_path = pathlib.Path("aclImdb")
    # The untarred directory is `aclImdb` instead of `aclImdb_v1`.

movie_ids = set()
for dataset in ["train", "test"]:
    p = data_path / dataset
    for p in p.iterdir():
        if p.name.startswith("urls"):
            print("Opening", p)
            with p.open() as f:
                for line in f:
                    movie_ids.add(line.split("/")[4])
print(len(movie_ids), list(movie_ids)[:2])

parser = argparse.ArgumentParser()
parser.add_argument("api_key", help="OMDb API key.")
parser.add_argument(
    "-test",
    action="store_true",
    help="Run on small set of movie IDs.",
)
args = parser.parse_args()


def get_id(movie_id):
    r = requests.get(
        "http://www.omdbapi.com",
        params={"i": movie_id, "apikey": args.api_key},
    )
    return r.json()["imdbID"]


if args.test:
    movie_ids = set(list(movie_ids)[:200])

id_map = {}
# API calls can fail, so after iterating over everything, back off then try again with the ones that failed.
back_off = 1
unknown_ids = []
while len(id_map) < len(movie_ids):
    ids_to_search = list(movie_ids - set(id_map.keys()))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=8
    ) as executor:
        # Start the load operations and mark each future with its URL
        future_to_id = {
            executor.submit(get_id, movie_id): movie_id
            for movie_id in ids_to_search
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_id),
            total=len(ids_to_search),
        ):
            movie_id = future_to_id[future]
            try:
                data = future.result()
            except Exception as exc:
                executor.shutdown(cancel_futures=True)
                print(
                    "%r generated an exception: %s" % (movie_id, exc)
                )
                import pdb

                pdb.set_trace()
                if exc.args == ("imdbID",):
                    unknown_ids.append(movie_id)
                    movie_ids.discard(movie_id)
                break
            else:
                id_map[movie_id] = data
    if len(id_map) < len(movie_ids):
        time.sleep(back_off)
        back_off *= 2

pd.DataFrame(
    [{"movie_id": k, "main_movie_id": v} for k, v in id_map.items()]
).to_csv("movie_ids.csv", index=False)
pd.DataFrame(unknown_ids, columns=["movie_id"]).to_csv(
    "unknown_movie_ids.csv", index=False
)
