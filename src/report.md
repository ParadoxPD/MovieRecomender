# Problem formulation

**To build a Movie Recommendations with Movielens Dataset**

### **What is a Recommendation System?**

Simply put a Recommendation System is a filtration program whose prime goal is to predict the “rating” or “preference” of a user towards a domain-specific item or item. In our case, this domain-specific item is a movie, therefore the main focus of our recommendation system is to filter and predict only those movies which a user would prefer given some data about the user him or herself.


### Collaborative Filtering
This filtration strategy is based on the combination of the user’s behavior and comparing and contrasting that with other users’ behavior in the database. The history of all users plays an important role in this algorithm. The main difference between content-based filtering and collaborative filtering that in the latter, the interaction of all users with the items influences the recommendation algorithm while for content-based filtering only the concerned user’s data is taken into account.
There are multiple ways to implement collaborative filtering but the main concept to be grasped is that in collaborative filtering multiple user’s data influences the outcome of the recommendation. and doesn’t depend on only one user’s data for modeling.

**There are 2 types of collaborative filtering algorithms:**

- ***User-based Collaborative filtering***:
The basic idea here is to find users that have similar past preference patterns as the user ‘A’ has had and then recommending him or her items liked by those similar users which ‘A’ has not encountered yet. This is achieved by making a matrix of items each user has rated/viewed/liked/clicked depending upon the task at hand, and then computing the similarity score between the users and finally recommending items that the concerned user isn’t aware of but users similar to him/her are and liked it.

    For example, if the user ‘A’ likes ‘Batman Begins’, ‘Justice League’ and ‘The Avengers’ while the user ‘B’ likes ‘Batman Begins’, ‘Justice League’ and ‘Thor’ then they have similar interests because we know that these movies belong to the super-hero genre. So, there is a high probability that the user ‘A’ would like ‘Thor’ and the user ‘B’ would like The Avengers’.

    **Disadvantages**

    - People are fickle-minded i.e their taste change from time to time and as this algorithm is based on user similarity it may pick up initial similarity patterns between 2 users who after a while may have completely different preferences.
    - There are many more users than items therefore it becomes very difficult to maintain such large matrices and therefore needs to be recomputed very regularly.
    - This algorithm is very susceptible to shilling attacks where fake users profiles consisting of biased preference patterns are used to manipulate key decisions.
    

- ***Item-based Collaborative Filtering***:
The concept in this case is to find similar movies instead of similar users and then recommending similar movies to that ‘A’ has had in his/her past preferences. This is executed by finding every pair of items that were rated/viewed/liked/clicked by the same user, then measuring the similarity of those rated/viewed/liked/clicked across all user who rated/viewed/liked/clicked both, and finally recommending them based on similarity scores.

    Here, for example, we take 2 movies ‘A’ and ‘B’ and check their ratings by all users who have rated both the movies and based on the similarity of these ratings, and based on this rating similarity by users who have rated both we find similar movies. So if most common users have rated ‘A’ and ‘B’ both similarly and it is highly probable that ‘A’ and ‘B’ are similar, therefore if someone has watched and liked ‘A’ they should be recommended ‘B’ and vice versa.

    **Advantages over User-based Collaborative Filtering**:
    - Unlike people’s taste, movies don’t change.
    - There are usually a lot fewer items than people, therefore easier to maintain and compute the matrices.
    - Shilling attacks are much harder because items cannot be faked.



# Algorithms followed



# Dataset used

[MovieLens Dataset](https://grouplens.org/datasets/movielens/) is used in this project to train the model for movie recommendations

# Results and Discussion

# References

- [Concepts](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf)
- [Hands-on recommendation system](https://realpython.com/build-recommendation-engine-collaborative-filtering/)
- [Dataset](https://grouplens.org/datasets/movielens/)
- [Movie Recommendation System](https://www.analyticsvidhya.com/blog/2020/11/create-your-own-movie-movie-recommendation-system/)
