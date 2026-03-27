from parameter_optimization import *
from surprise import SVD, KNNBasic
from fill_ratings_csv import *

def main():
    svd_model = SVD(
        n_factors=50,
        lr_all=0.01,
        reg_all=0.1,
        n_epochs=100,
        random_state=57
    )
    
    knn_model = KNNBasic(
        k=30,
        sim_options={
            'name': 'cosine',
            'user_based': True,
        },
        random_state=57
    )
    
    #grid_search_matrix_fact()
    
    #benchmark_precision_recall_RMSE(svd_model)
    
    #measure_computational_performance(svd_model)
    
    #measure_personalization(svd_model, 500)
    
    #train_and_get_prediction(svd_model, 7, 132333)
    
    generate_recommendations(svd_model)
    
    
if __name__ == "__main__":
    main()
