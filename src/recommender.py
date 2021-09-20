import pandas as pd
from model import Model
from data_preprocessing import DataPreprocessing


pd.set_option('display.max_columns', 20)


class Recommender:
    def __init__(self, dataset, movies, model):
        self.dataset = dataset
        self.movies = movies
        self.model = model

    def get_movie_recommendation(self, movie_name, csr_matrix, number_of_recommendations=10):
        try:
            movie_list = self.movies[self.movies['title'].str.lower().str.contains(
                movie_name.lower())]
            if len(movie_list):
                movie_idx = movie_list.iloc[0]['movieId']
                movie_idx = self.dataset[self.dataset['movieId']
                                         == movie_idx].index[0]
                distances, indices = self.model.kneighbors(
                    csr_matrix[movie_idx], n_neighbors=number_of_recommendations+1)
                rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(
                ), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
                recommend_frame = []
                for val in rec_movie_indices:
                    movie_idx = self.dataset.iloc[val[0]]['movieId']
                    idx = self.movies[self.movies['movieId']
                                      == movie_idx].index
                    recommend_frame.append(
                        {'Title': self.movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
                df = pd.DataFrame(recommend_frame, index=range(
                    number_of_recommendations, 0, -1))[::-1]["Title"]
                return df.values
            else:
                return "No movies found. Please check your input"
        except IndexError:
            return "No Recomendations Found"


class RecommendationSystem:

    def __init__(self, dataset_path):
        self.path = dataset_path
        self.data_processor = None
        self.model = None
        self.recommender = None

    def train_model(self):
        if not self.data_processor:
            self.preprocess_data()
        self.model = Model()
        self.model.train(self.data_processor.get_csr_matrix())
        self.model = self.model.get_model()

    def preprocess_data(self):
        self.data_processor = DataPreprocessing(self.path)
        self.data_processor.get_final_data()
        self.data_processor.scale_data()

    def recommend(self, movie_name):
        if not self.model:
            self.train_model()
        self.recommender = Recommender(self.data_processor.get_final_data(
        ), self.data_processor.get_movies_data(), self.model)
        return self.recommender.get_movie_recommendation(movie_name, self.data_processor.get_csr_matrix())
