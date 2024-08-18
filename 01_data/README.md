# Data Processing

In this chapter the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is cleaned. A few issues were raised about the dataset in the process.

- Duplicate reviews
- Train-test contamination
- Unknown number of review per movie

The relationship between movies and movie IDs is one-to-many, so each movie may have multiple movie IDs. The IMDB assigns one movie ID as the main ID for a movie and all other movie IDs for that movie redirect to that main ID. In order to better clean the data, the [OMDb API](https://www.omdbapi.com/) service was used to map all movie IDs to their movies associated main ID in `movie_data.py` resulting in a csv `movie_ids.csv`. This csv has two columns. `movie_id` which contains a movie ID seen in the IMDB dataset and `main_movie_id` which is the main movie ID for that movie.

There was one movie ID, tt0319477, in the test set that had no associated movie. `unknown_movie.py` extracted the reviews for this movie ID. After looking over the reviews, this movie ID is for the TV show [Hack](https://www.imdb.com/title/tt0320022/). I've put this movie ID mapping in `manual_movie_ids.csv`.

The csv files `movie_ids.csv` and `manual_movie_ids.csv` were combined with `combine_movie_id_files.py`.

