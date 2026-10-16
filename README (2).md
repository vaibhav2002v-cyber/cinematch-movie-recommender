# 🎬 CineMatch – Movie Recommender System

CineMatch is a content-based movie recommendation web app built with **Python** and **Streamlit**. Pick a movie you like and it suggests similar ones, complete with posters, ratings, genres, directors and a match percentage.

## ✨ Features

- 🔍 Searchable dropdown of ~4,800 movies from the TMDB 5000 dataset
- 🧠 Content-based filtering using cosine similarity
- 🖼️ Movie posters fetched live from the TMDB API (falls back to stylish placeholders if no key is provided)
- 🎚️ Adjustable number of recommendations (3 to 12)
- 🌙 Modern dark UI with hover animations and match-percentage badges

## 🧠 How It Works

1. Load and merge `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
2. Keep the useful columns: overview, genres, keywords, top 3 cast members and the director.
3. Combine them into a single `tags` text for each movie, lowercase it and apply Porter stemming.
4. Convert the tags into vectors with `CountVectorizer` (5,000 features, English stop words removed).
5. Compute the cosine similarity between all movies.
6. For the selected movie, return the top-N most similar movies.

## 🛠️ Tech Stack

- Python 3.9+
- Streamlit
- Pandas
- scikit-learn
- NLTK
- Requests
- TMDB API (for posters)

## 📁 Project Structure

```
├── movie_recommend.py        # Streamlit app
├── tmdb_5000_movies.csv      # Movie dataset
├── tmdb_5000_credits.csv     # Cast and crew dataset
├── Untitled.ipynb            # Notebook used for model exploration
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a TMDB API key (optional, for posters)

1. Create a free account at [themoviedb.org](https://www.themoviedb.org/).
2. Go to **Settings → API** and request an API key.
3. Provide the key in one of these ways (do **not** hard-code it in the source):
   - Paste it in the **TMDB API key** box in the app sidebar, or
   - Create `.streamlit/secrets.toml` with:

     ```toml
     TMDB_API_KEY = "your_api_key_here"
     ```

Without a key the app still works and shows placeholder posters.

### 5. Run the app

```bash
streamlit run movie_recommend.py
```

The app opens at `http://localhost:8501`.

## 📸 Usage

1. Type or select a movie from the dropdown.
2. Click **✨ Recommend movies**.
3. Browse the recommended movies, each showing its rating, year, director, genres and match percentage.

## 📊 Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle.

## 🔮 Future Improvements

- Hybrid recommendations (content + collaborative filtering)
- Filter by genre, year or rating
- Use TF-IDF or sentence embeddings instead of Count Vectorizer
- Deploy on Streamlit Community Cloud

## 🙏 Acknowledgements

- [TMDB](https://www.themoviedb.org/) for the data and poster API
  *(This product uses the TMDB API but is not endorsed or certified by TMDB.)*
- [Streamlit](https://streamlit.io/) for the web framework

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
