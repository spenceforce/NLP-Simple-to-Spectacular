import pandas as pd

known = pd.read_csv("movie_ids.csv")
manual = pd.read_csv("manual_movie_ids.csv")
assert set(known.columns) == set(manual.columns)
pd.concat([known, manual]).to_csv("all_movie_ids.csv", index=False)
