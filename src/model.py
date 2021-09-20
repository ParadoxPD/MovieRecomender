from sklearn.neighbors import NearestNeighbors


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
