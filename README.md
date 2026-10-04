# Creating Cohorts of Songs — Rolling Stones Spotify Analysis

Unsupervised machine learning project that clusters Rolling Stones songs into distinct cohorts using Spotify audio features, to support song recommendation use cases (similar to how Spotify powers personalized playlists for its 456M+ monthly users).

> Report generated with AI assistance.

## Objective

Perform exploratory data analysis and cluster analysis to group songs into cohorts based on audio characteristics, helping understand what features define similar types of songs for recommendation purposes.

## Dataset

- **1,610 Rolling Stones songs** pulled via the Spotify API (reduced to ~1,508 after cleaning)
- 11 standardized audio features: `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`, `popularity`, `duration_ms`
- Full field definitions in [`data/data_dictionary.xlsx`](data/data_dictionary.xlsx)

## Approach

1. **Data Cleaning** — Dropped unnamed index column, checked for duplicates and missing values (none found), parsed `release_date` into a `release_year` feature
2. **Feature Scaling** — Applied `StandardScaler` to all 11 audio features (loudness and tempo have very different ranges than 0–1 features like energy/acousticness, so scaling prevents them from dominating distance calculations)
3. **Outlier Removal** — Removed songs with any feature value beyond 3 standard deviations (Z-score), reducing the dataset from 1,610 → ~1,508 songs
4. **EDA** — Explored feature correlations, album-level popularity, and recording-type patterns
5. **Dimensionality Reduction (PCA)** — Found ~8 of 11 principal components explain 95% of variance; used for 2D cluster visualization
6. **Clustering (K-Means)** — Used the Elbow Method (tested k=2 to 12) to identify **k=4** as the optimal cluster count

## Key Findings

- **Energy and loudness are strongly correlated** (r = 0.76) — louder songs tend to be more energetic
- **Acousticness and energy are strongly negatively correlated** — acoustic tracks have lower energy
- **Popularity has no single strong predictor** — it depends on a combination of features
- Songs from the **1970s–80s** tend to have higher popularity today, reflecting classic catalog dominance
- **Live recordings form a naturally distinct cluster** (high liveness, high energy, long duration)

### The Four Song Cohorts

| Cluster | Label | Size | Defining Traits |
|---|---|---|---|
| 0 | Popular High-Energy Studio Hits | 356 songs | Highest popularity (~27.3), loudest (-4.9 dB), high valence (0.74) |
| 1 | Mellow Acoustic / Chill Studio Tracks | 262 songs | Highest acousticness (0.48), lowest energy, melancholic tone (valence 0.39) |
| 2 | Live High-Energy Rock Tracks | 520 songs | Highest liveness (0.84), fastest tempo (137 BPM), longest duration (280s) |
| 3 | Groovy Upbeat Studio Tracks | 361 songs | Most danceable (0.60), most upbeat (valence 0.75), shortest duration (187s) |

**Top recommended albums** (by count of popular songs, score ≥ 40): *Sticky Fingers (Remastered)* and *Exile On Main Street (2010 Re-Mastered)*.

Full methodology, cluster profiles, and visualizations in [`reports/Rolling_Stones_Cohort_Analysis_Report.pdf`](reports/Rolling_Stones_Cohort_Analysis_Report.pdf).

## Recommendations

- Use cohort labels as a feature in collaborative filtering models to improve cold-start recommendations
- Serve Cluster 0 (High-Energy Hits) to new users as a default onboarding entry point
- Build mood-based playlists: Cluster 1 for late-night listening, Cluster 3 for parties, Cluster 2 for concert/live-music fans
- Retrain clusters periodically as new albums and streaming data become available
- Extend the analysis to the full Spotify catalog for cross-artist recommendations

## Tech Stack

- **Python 3**, Jupyter Notebook
- `pandas`, `numpy` for data manipulation
- `scikit-learn` — `StandardScaler`, `KMeans`, `PCA`
- `matplotlib`, `seaborn` for visualization (elbow curve, correlation heatmap, PCA scatter plots)

## Project Structure

```
├── notebooks/
│   └── Creating_Cohorts_of_Songs.ipynb        # Full EDA, preprocessing & clustering pipeline
├── reports/
│   └── Rolling_Stones_Cohort_Analysis_Report.pdf  # Final report with cluster definitions & insights
├── data/
│   ├── rolling_stones_spotify.csv              # Raw Spotify audio feature dataset
│   └── data_dictionary.xlsx                     # Feature definitions
├── docs/
│   └── problem_statement.docx                   # Original course project brief
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook notebooks/Creating_Cohorts_of_Songs.ipynb
```

## Author

Sanjay Gosai
*Machine Learning Course-End Project*
