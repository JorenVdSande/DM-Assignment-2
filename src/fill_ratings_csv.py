import pandas as pd
from surprise import SVD, Dataset, Reader
import numpy as np

"""
Pandas data type for 'ratings_train.csv'
"""
d_type_ratings_train = {
    "userId": "uint16",
    "movieId": "uint32",
    "rating": "float",
    "timestamp": "uint32"
}

d_type_ratings_test = {
    "userId": "uint16",
    "movieId": "uint32",
    "rating": "float",
    "timestamp": "uint32"
}


def generate_recommendations(model):
    """
    Fill the 'ratings_test.csv' file with the 10 best recommendations according to the given model.
    
    :param model: The SVD model
    """
    np.random.seed(57)
    
    ratings_train = pd.read_csv("data/ratings_train.csv", dtype=d_type_ratings_train)
    ratings_test = pd.read_csv("data/ratings_test.csv")
    
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_train[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model.fit(trainset)
    
    all_movie_ids = ratings_train['movieId'].unique()
    known_users = set(ratings_train['userId'].unique())
    
    k = 10
    
    # popularity fallback for cold start
    popularity = ratings_train.groupby('movieId').agg(
        count=('rating', 'count'),
        mean_rating=('rating', 'mean')
    )
    popularity['score'] = popularity['count'] * popularity['mean_rating']
    popular_movies = list(popularity.nlargest(k, 'score').index)
    
    results = []
    
    for _, row in ratings_test.iterrows():
        uid = row['userId']
        
        if uid not in known_users:
            # cold start --> return popular movies
            recommendations = popular_movies[:k]
        else:
            # get movies this user has NOT already rated
            rated_movies = set(ratings_train[ratings_train['userId'] == uid]['movieId'])
            unrated_movies = [iid for iid in all_movie_ids if iid not in rated_movies]
            
            # predict ratings for all unrated movies
            pairs = [(uid, iid, 0) for iid in unrated_movies]
            predictions = model.test(pairs)
            
            # sort by predicted rating and take top-K
            predictions.sort(key=lambda x: x.est, reverse=True)
            recommendations = [pred.iid for pred in predictions[:k]]
        
        result = {'userId': int(uid)}
        for i, movie_id in enumerate(recommendations, 1):
            result[f'recommendation{i}'] = movie_id
        results.append(result)
    
    output = pd.DataFrame(results)
    output.to_csv("generated_data/ratings_test.csv", index=False)