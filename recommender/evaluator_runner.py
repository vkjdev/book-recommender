from evaluator2 import Evaluator

from dummy_recommender import MeanRatingRecommender as Recommender
from distance_recommender import UserDistanceRecommender as Recommender
# from doc2vec_recommender import Doc2VecRecommender as Recommender

# PICKLE_USER_RANGE = [50, 50]
#
# PICKLE_DATA = {"train_data": "/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/%s_%s_ratings_train.dat"
#                              % (PICKLE_USER_RANGE[0], PICKLE_USER_RANGE[1]),
#                "test_data": "/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/%s_%s_ratings_test.dat"
#                             % (PICKLE_USER_RANGE[0], PICKLE_USER_RANGE[1])}

# param for choosing the pickled data as exported from data_pickler.py
USERS_TO_COLLECT = 200
# use none to evaluate on all data in testing dataframe
TESTED_DATA_SIZE = None

PICKLE_DATA = {"train_data": "/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/%s_users_ratings_train.dat"
                             % USERS_TO_COLLECT,
               "test_data": "/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/%s_users_ratings_test.dat"
                            % USERS_TO_COLLECT}

tested_recommender = Recommender()
evaluator = Evaluator(pickle_boost=True)
score = evaluator.evaluate(tested_recommender, pickled_train_filepath=PICKLE_DATA["train_data"], pickled_test_filepath=PICKLE_DATA["test_data"], tested_volume=TESTED_DATA_SIZE)

print("DONE")
print("Score: %s" % score)
