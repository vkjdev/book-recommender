# validation method for recommender systems
# computes the RMSE (root mean squared error) for the recommender method on the same dataset
# using increasing volume ratio of training/testing dataset
# might be used to evaluate how the method improves on increasing volume on testing dataset

import logging
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
# from dummy_recommender import MeanRatingRecommender as Recommender
from knn_recommender_v2 import KnnRecommender as Recommender
# from doc2vec_recommender import Doc2VecRecommender as Recommender

import cPickle

PICKLE_BOOST = {"data_select": True,
                "train_data": True,
                "test_data": True}
PICKLE_DATA = "/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/20_60_ratings_data.dat"

DATA_FILE_PATH="/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/ratings_Books.csv"
# DATA_FILE_PATH = '/home/kvassay/data/book-recommender/ratings_Books.csv'

# slices to use for testing methods improvements on increasing amount of testing data
SLICING_INTERVAL = 5

# select how many times the evaluation will split data and test
# selecting 1 means one split with fold on SLICING_INTERVAL-1/SLICING_INTERVAL timestamp for every user
# can automatically test a development of model performance on increasing amount of training data
SLICING_RUNS = 1

method_name = Recommender.__module__

# logging init:
logger = logging.getLogger()
logger.setLevel(20)

FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

data_file = DATA_FILE_PATH

date_format = "%Y-%m-%d %H:%M:%S"

logger.info("Starting application on %s dataset" % data_file)
logger.info("Testing recommender implementation of %s " % method_name)

# TODO: pohrat si s filtrovanim dat, evaulaciou
# TODO: Na prezentaciu: popis dat, popis pouzitych metod, evaulacia - metodika, vysledky
# TODO: profiling - zryclit vyber dat userov, zvysit pocet userov

if PICKLE_BOOST["data_select"]:
    logger.info("Pickle boost on")

    with open(PICKLE_DATA, "r") as pickle_file:
        eval_dataframe = cPickle.load(pickle_file)
    eval_users = eval_dataframe["user"].unique()

else:
    SAMPLED_USERS = 1000
    USER_IS_ROBOT_THRESHOLD = 100

    logger.info("Pickle boost off - serialize data using data_pickler.py, set the pickle output, and set PICKLE_BOOST on to speed up loading")
    with open(data_file, "r") as f:
        df = pd.read_csv(data_file)

    grouped_users = df.groupby(["user"]).count()
    user_data = grouped_users[grouped_users["item"] >= SLICING_INTERVAL]
    n_review_users = grouped_users[grouped_users["item"] <= USER_IS_ROBOT_THRESHOLD]["item"].keys()
    eval_users = np.random.choice(n_review_users.unique(), SAMPLED_USERS)
    logger.info("Selected %s users to evaluate their ratings" % eval_users.__len__())

    eval_dataframe = df[df['user'].isin(eval_users)]
    logger.info("Selected dataframe of random %s users containing %s entries" %
                (eval_users.__len__(), eval_dataframe.__len__()))

logger.info("Eval data loaded")

for eval_run in range(SLICING_INTERVAL - SLICING_RUNS, SLICING_INTERVAL):
    # each test run starts with clean recommender instance
    tested_recommender = Recommender()

    training_frame = pd.DataFrame(columns=["user", "item", "rating", "timestamp"])
    testing_frame = pd.DataFrame(columns=["user", "item", "rating", "timestamp"])

    quantile = eval_run * (1 / float(SLICING_INTERVAL))
    logger.info("training on users dataset divides on quantile %s" % quantile)

    # TODO: later compare recommender results with using dataset having only data newer than from 2013

    for user in eval_users:
        user_reviews = eval_dataframe[eval_dataframe['user'] == user]

        # value dividing reviews of a user to training and testing
        slicing_timestamp = user_reviews["timestamp"].quantile(q=quantile)

        training_user_data = user_reviews[user_reviews["timestamp"] < slicing_timestamp]
        training_frame = training_frame.append(training_user_data)

        testing_user_data = user_reviews[user_reviews["timestamp"] >= slicing_timestamp]
        testing_frame = testing_frame.append(testing_user_data)

    logger.info("training dataframe size: %s" % training_frame.__len__())
    logger.info("testing dataframe size: %s" % testing_frame.__len__())

    # TODO: choose to train on pandas dataframe, or raw csv
    # tested_recommender.fit(training_frame.to_csv())
    tested_recommender.fit(training_frame)
    logger.info("Recommender method has fit on %s entries" % training_frame.__len__())

    # aggregated difference of recommender predicted rating against the real rating
    y_true=list()
    y_pred=list()
    len_diff = 0
    delta_sum = 0
    for index, entry in testing_frame.iterrows():
        expected_score = entry["rating"]
        actual_score = tested_recommender.predict(entry["user"], entry["item"])

        # TODO: remove for no output of matching
        logger.info("expected - actual: %s - %s" % (expected_score, actual_score))

        if actual_score is not None:
            delta_sum += math.fabs(expected_score - actual_score)
            y_true.append(expected_score)
            y_pred.append(actual_score)
        else:
            len_diff += 1

    logger.info("Recommender method has predicted %s ratings" % (testing_frame.__len__()-len_diff))
    logger.info('Recommender failed to predict %s ratings', len_diff)

    # mae = float(delta_sum) / (testing_frame.__len__() - len_diff)
    mae = mean_absolute_error(y_true,y_pred)

    mean_rating = float(testing_frame["rating"].mean())

    logger.info("Testing data mean rating: %s" % mean_rating)
    logger.info("")
    logger.info("Method %s mean absolute error %s" % (method_name, mae))
    logger.info("")
