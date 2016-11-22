# validation method for recommender systems
# computes the RMSE (root mean squared error) for the recommender method on the same dataset
# using increasing volume ratio of training/testing dataset
# might be used to evaluate how the method improves on increasing volume on testing dataset

import logging
import math
import pandas as pd
import numpy as np
from dummy_recommender import MeanRatingRecommender as Recommmender
# from knn_recommender_v2 import KnnRecommender as Recommmender

SAMPLED_USERS = 1000
USER_IS_ROBOT_THRESHOLD = 1000

# slices to use for testing methods improvements on increasing amount of testing data
SLICING_INTERVAL = 5

# select how many times the evaluation will split data and test
# selecting 1 means one split with fold on SLICING_INTERVAL-1/SLICING_INTERVAL timestamp for every user
# can automatically test a development of model performance on increasing amount of training data
SLICING_RUNS = 1

method_name = Recommmender.__module__

# logging init:
logger = logging.getLogger()
logger.setLevel(20)

FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

data_file = "/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/ratings_Books.csv"

date_format = "%Y-%m-%d %H:%M:%S"

logger.info("Starting application on %s dataset" % data_file)
logger.info("Testing recommender implementation of %s " % method_name)

with open(data_file, "r") as f:

    df = pd.read_csv(data_file)
    grouped_users = df.groupby(["user"]).count()
    n_review_users = grouped_users[grouped_users["item"] >= SLICING_INTERVAL]["item"].keys()
    eval_users = np.random.choice(n_review_users.unique(), SAMPLED_USERS)
    logger.info("Selected %s users to evaluate their ratings" % eval_users.__len__())

    eval_dataframe = df[df['user'].isin(eval_users)]
    logger.info("Selected dataframe of random %s users containing %s entries" %
                (eval_users.__len__(), eval_dataframe.__len__()))

    for eval_run in range(SLICING_INTERVAL-SLICING_RUNS, SLICING_INTERVAL):
        # each test run starts with clean recommender instance
        tested_recommender = Recommmender()

        training_frame = pd.DataFrame(columns=["user", "item", "rating", "timestamp"])
        testing_frame = pd.DataFrame(columns=["user", "item", "rating", "timestamp"])

        quantile = eval_run*(1/float(SLICING_INTERVAL))
        logger.info("training on users dataset divides on quantile %s" % quantile)

        # TODO: later compare recommender results with using dataset having only data newer than from 2013

        for user in eval_users:
            user_reviews = eval_dataframe[eval_dataframe['user'] == user]

            if len(user_reviews) >= USER_IS_ROBOT_THRESHOLD:
                # do not include users having more than threshold ratings
                continue

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
        delta_sum = 0
        for index, entry in testing_frame.iterrows():
            expected_score = entry["rating"]
            actual_score = tested_recommender.predict(entry["user"], entry["item"])

            # TODO: remove for no output of matching
            logger.info("expected - actual: %s - %s" % (expected_score, actual_score))

            delta_sum += math.fabs(expected_score-actual_score)

        logger.info("Recommender method has predicted %s ratings" % testing_frame.__len__())

        delta = float(delta_sum)/testing_frame.__len__()
        mean_rating = float(testing_frame["rating"].mean())

        logger.info("Testing data mean rating: %s" % mean_rating)
        logger.info("")
        logger.info("Method %s average error delta %s" % (method_name, delta))
        logger.info("")
