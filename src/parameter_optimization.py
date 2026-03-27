import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV, cross_validate, train_test_split
from collections import defaultdict
import numpy as np
import random
import time

"""
Pandas data type for 'ratings_train.csv'
"""
d_type_ratings_train = {
    "userId": "uint16",
    "movieId": "uint32",
    "rating": "float",
    "timestamp": "uint32"
}

def grid_search_matrix_fact():
    """
    Search a grid of values and find which combination results in the
    most optimal RMSE for the SVD model.
    """
    random.seed(57)
    np.random.seed(57)
    
    ratings_train = pd.read_csv("data/ratings_train.csv", dtype=d_type_ratings_train)
    
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_train[['userId', 'movieId', 'rating']], reader)
    
    print("Starting Grid Search...")
    
    param_grid = {
        'n_factors': [5, 10, 20, 50, 100],
        'lr_all': [0.002, 0.005, 0.01, 0.02],
        'reg_all': [0.02, 0.05, 0.1, 0.2],
        'n_epochs': [20, 50, 100],
        'random_state': [57],
    }
    
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    
    best_dim = gs.best_params['rmse']['n_factors']
    best_lr = gs.best_params['rmse']['lr_all']
    best_reg = gs.best_params['rmse']['reg_all']
    best_epochs = gs.best_params['rmse']['n_epochs']
    print(f"Best RMSE score: {gs.best_score['rmse']:.4f}")
    print(f"Optimal number of latent dimensions: {best_dim}")
    print(f"Best learning rate: {best_lr}")
    print(f"Best regularization rate: {best_reg}")
    print(f"Best number of epochs: {best_epochs}")
    
    # build model using the best parameters
    best_model = gs.best_estimator['rmse']
    trainset = data.build_full_trainset()
    best_model.fit(trainset)
    
    # make a prediction
    user_id = 1
    movie_id = 1
    pred = best_model.predict(user_id, movie_id)
    print(f"Predicted rating for User {user_id} on Item {movie_id}: {pred.est:.2f}")

def get_precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Compute Precision@K and Recall@K for each user, then average across all users.

    A 'relevant' movie is one whose TRUE rating is >= threshold.
    A 'recommended' movie is one that appears in the top-K predictions.
    A 'hit' is a relevant movie that also appears in the top-K.
    
    :param predictions: List of Surprise prediction objects
    :param k: Number of top items to consider
    :param threshold: Minimum true rating to consider an item relevant
    """
    # Group predictions by user
    user_predictions = defaultdict(list)
    for pred in predictions:
        user_predictions[pred.uid].append((pred.est, pred.r_ui))
    
    precisions, recalls = [], []
    
    for uid, user_preds in user_predictions.items():
        # skip users who don't have enough ratings to fill top-K
        if len(user_preds) < k:
            continue
            
        # sort by estimated rating and take top-K
        user_preds.sort(key=lambda x: x[0], reverse=True)
        top_k = user_preds[:k]
        
        # Count relevant items in the FULL prediction list (not just top-K)
        n_relevant = sum(true_r >= threshold for _, true_r in user_preds)
        
        # Count hits: items in top-K that are also relevant
        n_hits = sum(true_r >= threshold for _, true_r in top_k)
        
        precisions.append(n_hits / k)  # -> Precision@K
        recalls.append(n_hits / n_relevant if n_relevant > 0 else 0)  # -> Recall@K
    
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    
    return avg_precision, avg_recall


def count_evaluated_users(predictions, k):
    """
    Helper to see how many users actually contributed to Precision/Recall.
    """
    user_predictions = defaultdict(list)
    for pred in predictions:
        user_predictions[pred.uid].append(pred)
    return sum(1 for preds in user_predictions.values() if len(preds) >= k)


def benchmark_precision_recall_RMSE(model):
    """
    Calculate the RMSE, the precision@k and recall@k.
    k has been set to 10, and te threshold is 4.0
    
    :param model: a model from the surprise library like SVD or KNNBasic
    """
    np.random.seed(57)
    
    ratings_train = pd.read_csv("data/ratings_train.csv", dtype=d_type_ratings_train)
    
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_train[['userId', 'movieId', 'rating']], reader)
    
    # keep 80% for training, 20% for ranking evaluation
    trainset, testset = train_test_split(data, test_size=0.2, random_state=57)
    
    cv_data = Dataset.load_from_df(
        ratings_train.sample(frac=0.8, random_state=57)[['userId', 'movieId', 'rating']],
        reader
    )
    cv_results = cross_validate(model, cv_data, measures=['RMSE'], cv=5, verbose=False)
    mean_rmse = cv_results['test_rmse'].mean()
    print(f"Cross-validated RMSE: {mean_rmse:.4f}")
    
    # calculate precision and recall
    model.fit(trainset)
    predictions = model.test(testset)
    
    k = 10
    threshold = 4.0
    precision, recall = get_precision_recall_at_k(predictions, k=k, threshold=threshold)
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}:    {recall:.4f}")
    print(f"Users evaluated: {count_evaluated_users(predictions, k)}")
    
    # retrain on full data for actual use
    full_trainset = data.build_full_trainset()
    model.fit(full_trainset)
    
    # make a prediction
    user_id = 1
    movie_id = 2338
    prediction = model.predict(user_id, movie_id)
    print(f"Predicted rating for user {user_id} on movie {movie_id}: {prediction.est:.2f}")
    
    
def measure_computational_performance(model):
    """
    Time how long the model takes to train and how long it takes to make a number of predictions
    :param model: a model from the surprise library like SVD or KNNBasic
    """
    np.random.seed(57)
    
    ratings_train = pd.read_csv("data/ratings_train.csv", dtype=d_type_ratings_train)
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_train[['userId', 'movieId', 'rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.2, random_state=57)
    
    # time training
    start = time.time()
    model.fit(trainset)
    train_time = time.time() - start
    print(f"Training time: {train_time:.2f}s")
    
    # time predictions —> measured on a batch for reliability
    start = time.time()
    predictions = model.test(testset)
    predict_time = time.time() - start
    print(f"Prediction time ({len(testset)} predictions): {predict_time:.2f}s")
    
    
def measure_personalization(model, k=10):
    """
    Measures what fraction of each user's top-K recommendations
    are NOT in the globally most popular K movies.
    Higher = more personalised.
    
    :param model: a model from the surprise library like SVD or KNNBasic
    :param k: the top-K size you want to look at
    """
    ratings_train = pd.read_csv("data/ratings_train.csv", dtype=d_type_ratings_train)

    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_train[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model.fit(trainset)  # model now has a trainset attribute

    all_user_ids = ratings_train['userId'].unique()
    all_movie_ids = ratings_train['movieId'].unique()

    all_pairs = [(uid, iid, 0) for uid in all_user_ids for iid in all_movie_ids]
    predictions = model.test(all_pairs)

    user_predictions = defaultdict(list)
    for pred in predictions:
        user_predictions[pred.uid].append((pred.est, pred.iid))

    user_topk = {}
    for uid, preds in user_predictions.items():
        preds.sort(key=lambda x: x[0], reverse=True)
        user_topk[uid] = set(iid for _, iid in preds[:k])
    
    popularity = ratings_train.groupby('movieId').agg(
        count=('rating', 'count'),
        mean_rating=('rating', 'mean')
    )
    
    popularity['score'] = popularity['count'] * popularity['mean_rating']
    global_popular = set(popularity.nlargest(k, 'score').index)

    scores = []
    for uid, topk in user_topk.items():
        unique_to_user = len(topk - global_popular) / k
        scores.append(unique_to_user)

    result = sum(scores) / len(scores)
    print(f"Model personalization score: {result:.4f}")
    
    
def train_and_get_prediction(model, user_id=1, movie_id=1):
    """
    Train the model and do make a prediction for user_id and movie_id
    
    :param model: a model from the surprise library like SVD or KNNBasic
    :param user_id: The user_id for which you want to make a rating prediction
    :param movie_id: The movie_id for which you want to make a rating prediction
    """
    ratings_train = pd.read_csv("data/ratings_train.csv", dtype=d_type_ratings_train)

    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_train[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model.fit(trainset)
    
    prediction = model.predict(user_id, movie_id)
    print(f"Predicted rating for user {user_id} on movie {movie_id}: {prediction.est:.2f}")
    