from recommender import RecommendationSystem as RM
import os


if __name__ == '__main__':
    dataset_path = os.path.abspath('../Dataset/data/')

    rm = RM(dataset_path)
    print("\n".join(rm.recommend("Iron Man")))
