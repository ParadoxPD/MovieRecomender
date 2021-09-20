import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler


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
            # importing the dataset
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
