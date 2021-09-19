import pandas as pd
# import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', 20)


class DataPreprocessing:
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.dataset = None
        self.data_processed = None
        self.data_processed_scaled = None
        self.ratings = None
        self.movies = None

    def import_dataset(self):
        if self.dataset == None:
            #importing the dataset
            self.ratings = pd.read_csv(self.path + '/ratings.csv')
            self.movies = pd.read_csv(self.path + '/movies.csv')
            self.dataset = self.ratings.merge(self.movies)

    def process_data(self):
        # Data cleaning
        if self.ratings is None:
            self.import_dataset()
        ratings_ = self.ratings.drop(['timestamp'], axis=1)
        no_movies_voted = ratings_.groupby('userId')['rating'].agg('count')
        ratings_ = ratings_.loc[no_movies_voted[no_movies_voted > 10].index, :]
        return ratings_

    def get_movies_data(self):
        return self.movies

    def get_final_data(self):
        # creating the final dataset containing the movies ,user and their ratings
        self.data_processed = self.process_data()
        self.data_processed = self.data_processed.pivot(
            index='movieId', columns='userId', values='rating')
        self.data_processed.fillna(0, inplace=True)
        self.data_processed.reset_index(inplace=True)
        return self.data_processed

    def scale_data(self):
        # Scaling the ratings matrix using Standard Scaler
        scaler = StandardScaler(with_mean=False)
        self.data_processed_scaled = scaler.fit_transform(self.data_processed)
        return self.data_processed_scaled

    def get_csr_matrix(self):
        # Removing Sparsity
        try:
            csr_data = csr_matrix(self.data_processed_scaled)
            return csr_data
        except Exception:
            pass


class Model:

    # Using K Nearest Neighbours to find the recomendations using the similarity betwen the movies
    def __init__(self):
        self.model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        self.train_data = None

    def train(self, train_data):
        self.train_data = train_data
        self.model.fit(self.train_data)

    def get_model(self):
        return self.model


class Recommender:
        def __init__(self, dataset, movies, model):
            self.dataset = dataset
            self.movies = movies
            self.model = model

        
        def get_movie_recommendation(self, movie_name, csr_matrix, number_of_recommendations= 10):
            try:
                movie_list = self.movies[self.movies['title'].str.contains(
                    movie_name)]
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
