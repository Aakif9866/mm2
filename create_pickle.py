import pickle
from preprocessing.build_pipeline import build_movie_recommender
import os

os.makedirs("artifacts", exist_ok=True)

movies, similarity = build_movie_recommender()

with open("artifacts/movie_list.pkl", "wb") as f:
    pickle.dump(movies, f)

with open("artifacts/similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print("✅ Pickles generated successfully!")
