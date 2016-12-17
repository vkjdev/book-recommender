from evaluatorClass import Evaluator

# from dummy_recommender import MeanRatingRecommender as Recommender
from distance_recommender import UserDistanceRecommender as Recommender
# from doc2vec_recommender import Doc2VecRecommender as Recommender
# from similar_rating_book_recommender import SimilarRatingBookRecommender as Recommender

# param for choosing the pickled data as exported from data_pickler.py
USERS_TO_COLLECT = 1000
# use none to evaluate on all data in testing dataframe
TESTED_DATA_SIZE = None

PICKLE_DATA = {"train_data": "/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/%s_users_ratings_train.dat"
                             % USERS_TO_COLLECT,
               "test_data": "/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/%s_users_ratings_test.dat"
                            % USERS_TO_COLLECT}

tested_recommender = Recommender()
evaluator = Evaluator(pickle_boost=True)

TRAINING_USERS = 985
TESTED_VOLUME = 1000

NO_INCREASES = 1

results_array = []
for dataset_iter in range(NO_INCREASES):
    current_training_users = TRAINING_USERS / NO_INCREASES * (dataset_iter + 1)
    current_tested_users = TESTED_VOLUME/NO_INCREASES*(dataset_iter+1)
    score = evaluator.evaluate(tested_recommender, pickled_train_filepath=PICKLE_DATA["train_data"],
                               pickled_test_filepath=PICKLE_DATA["test_data"], tested_samples=current_tested_users,
                               training_users=current_training_users)
    results_array.append(score)
    print("DONE %s dataset increment" % dataset_iter)
    print("Score: %s" % score)

print("All DONE")
print("gathered results:\n%s" % results_array)
