from recommender import RecommendationSystem as RM
import os
import sys

if __name__ == '__main__':
    dataset_path = os.path.abspath('../Dataset/data/')

    movie = " ".join(sys.argv[1:])

    rm = RM(dataset_path)
    recommendations = rm.recommend(movie if movie else "Iron Man")
    print("Recommendations :", end="\n")
    print("\n".join(recommendations))
