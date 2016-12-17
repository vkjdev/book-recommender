from evaluator2 import Evaluator

from similar_rating_book_recommender import SimilarRatingBookRecommender as Recommender
#from knn_recommender_v2 import KnnRecommender as Recommender
#from knn_recommender_v3 import KnnRecommender as Recommender
# from doc2vec_recommender import Doc2VecRecommender as Recommender

PICKLE_USER_RANGE = [200, 200]

PICKLE_DATA = {"train_data": "200_users_ratings_train.dat",
              "test_data": "200_users_ratings_test.dat"}

tested_recommender = Recommender()
evaluator = Evaluator(pickle_boost=True)
score = evaluator.evaluate(tested_recommender, pickled_train_filepath=PICKLE_DATA["train_data"], pickled_test_filepath=PICKLE_DATA["test_data"])

print("DONE")
print("Score: %s" % score)