import pickle 
import streamlit as st 
import requests
import os
from dotenv import load_dotenv 
from preprocessing.build_pipeline import build_movie_recommender

load_dotenv()
api_key = os.getenv("API_KEY")


if not os.path.exists("artifacts/movie_list.pkl") or not os.path.exists("artifacts/similarity.pkl"):
    st.warning("Generating model files... This may take a minute ⏳")
    os.makedirs("artifacts", exist_ok=True)
    movies, similarity = build_movie_recommender()
    pickle.dump(movies, open("artifacts/movie_list.pkl", "wb"))
    pickle.dump(similarity, open("artifacts/similarity.pkl", "wb"))
else:
    movies = pickle.load(open("artifacts/movie_list.pkl", "rb"))
    similarity = pickle.load(open("artifacts/similarity.pkl", "rb"))


st.header("🎬 Movie Recommendation System")


movies = pickle.load(open('artifacts/movie_list.pkl', 'rb'))
similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie", movie_list)

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    if response.status_code != 200:
        return "https://via.placeholder.com/500x750?text=No+Image"
    data = response.json()
    poster_path = data.get('poster_path')
    return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Image"

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    rec_names, rec_posters = [], []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        rec_names.append(movies.iloc[i[0]].title)
        rec_posters.append(fetch_poster(movie_id))
    return rec_names, rec_posters

if st.button("Show Recommendation"):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(names[i])
            st.image(posters[i])






# old file
# import pickle 
# import streamlit as st 
# import requests
# import gdown
# import os
# from dotenv import load_dotenv 

# load_dotenv()


# # both pkl files together go more than 280 mb
# # so not possible to git push
# # creating a pipeline is better option but not time
# # so i'll download the files using gdown


# def download_artifacts():
#     os.makedirs('artifacts', exist_ok=True)

#     files = {
#         "movie_list.pkl": "1rgY8bm5_wS3bUgmwKkeQD5tOx6Bn0x2b",
#         "similarity.pkl": "1Y6QgLdfA43ab1meMdu75oGqMnWzIhE8v"
#     }

#     for filename, file_id in files.items():
#         path = f'artifacts/{filename}'
#         if not os.path.exists(path):
#             url = f'https://drive.google.com/uc?id={file_id}'
#             print(f"Downloading {filename}...")
#             gdown.download(url, path, quiet=False)

# # 🔽 Download the artifacts at app start
# download_artifacts()

# api_key = os.getenv("API_KEY")



# st.header("Movie Recommendation System Using Machine Learning")

# # Load data
# movies = pickle.load(open('artifacts/movie_list.pkl','rb'))
# similarity = pickle.load(open('artifacts/similarity.pkl','rb'))

# movie_list = movies['title'].values

# selected_movie = st.selectbox(
#     'Type or select movie to get recommendation',
#     movie_list
# )

# def fetch_poster(movie_id):
#     url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
#     response = requests.get(url)
#     if response.status_code != 200:
#         return "https://via.placeholder.com/500x750?text=No+Image"
#     data = response.json()
#     poster_path = data.get('poster_path')
#     if not poster_path:
#         return "https://via.placeholder.com/500x750?text=No+Image"
#     full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#     return full_path

# def recommend(movie):
#     index = movies[movies['title'] == movie].index[0]
#     distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
#     # [(5, 1.0), (23, 0.83), (12, 0.79), ...] -->> this is movie output
#     # movie id followed by score
#     recommended_movie_name = []
#     recommended_movie_poster = []
#     for i in distances[1:6]:  
#         movie_id = movies.iloc[i[0]].movie_id
#         recommended_movie_name.append(movies.iloc[i[0]].title)
#         recommended_movie_poster.append(fetch_poster(movie_id))
#     return recommended_movie_name, recommended_movie_poster


# if st.button('Show Recommendation'):
#     recommended_movie_name, recommended_movie_poster = recommend(selected_movie)
#     cols = st.columns(5)
#     for idx, col in enumerate(cols):
#         with col:
#             st.text(recommended_movie_name[idx])
#             st.image(recommended_movie_poster[idx])

