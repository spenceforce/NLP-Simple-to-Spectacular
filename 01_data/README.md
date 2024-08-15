# Data Processing

In this chapter the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is cleaned. A few issues were raised about the dataset in the process.

- Duplicate reviews
- Train-test contamination
- Unknown number of review per movie

The relationship between movies and movie IDs is one-to-many, so each movie may have multiple movie IDs. In order to better clean the data, the [OMDb API](https://www.omdbapi.com/) service was used to map all movie IDs to their canonical form. See `movie_data.py`.

There was one movie ID, tt0319477, in the test set that had no associated movie. `unknown_movie.py` extracted the reviews for this movie ID. After looking over the reviews, this movie ID is for the TV show [Hack](https://www.imdb.com/title/tt0320022/). I've put this movie ID mapping in `manual_movie_ids.csv`.
