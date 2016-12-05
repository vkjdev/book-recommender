# validation class for recommender systems
# computes the selected metrics for the recommender method on the same dataset

import cPickle
import logging
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dummy_recommender import MeanRatingRecommender as Recommender
# from knn_recommender_v2 import KnnRecommender as Recommender
# from doc2vec_recommender import Doc2VecRecommender as Recommender

# slices to use for testing methods improvements on increasing amount of testing data
# NOT SUPPORTED
# SLICING_INTERVAL = 5

# select how many times the evaluation will split data and test
# selecting 1 means one split with fold on SLICING_INTERVAL-1/SLICING_INTERVAL timestamp for every user
# can automatically test a development of model performance on increasing amount of training data
# NOT SUPPORTED
# SLICING_RUNS = 1

# logging init:
logger = logging.getLogger()
logger.setLevel(20)

FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

date_format = "%Y-%m-%d %H:%M:%S"

# DONE: od Pelanka: pohrat si s filtrovanim pouzitelnejsich dat, evaulaciou
# TODO: Na prezentaciu: popis dat, popis pouzitych metod, evaulacia - metodika, vysledky, vizualizacia
# DONE: profiling - zryclit vyber dat userov, zvysit pocet userov


class Evaluator:
    eval_methods = ["mae", "rmse", "precision", "recall", "f_score"]

    eval_dataframe = None
    eval_users = None

    training_frame = None
    testing_frame = None

    y_true = None
    y_pred = None

    _pickle_boost = None

    load = False
    split = False

    def __init__(self, pickle_boost=True):
        self._pickle_boost = pickle_boost

    def load_data(self, data_filepath, sampled_users=1000, min_ratings=5, user_is_robot_threshold=100):

        logger.info("Loading data from %s" % data_filepath)
        with open(data_filepath, "r") as f:
            df = pd.read_csv(f)
        logger.info("Loaded. Filtering data to users with # ratings in [%s, %s]" % (user_is_robot_threshold, sampled_users))
        grouped_users = df.groupby(["user"]).count()
        self.eval_users = grouped_users[grouped_users["item"] >= min_ratings]
        n_review_users = grouped_users[grouped_users["item"] <= user_is_robot_threshold]["item"].keys()
        eval_users = np.random.choice(n_review_users.unique(), sampled_users)
        logger.info("Selected %s users to evaluate their ratings" % eval_users.__len__())

        eval_dataframe = df[df['user'].isin(eval_users)]
        logger.info("Selected dataframe of %s users containing %s entries" %
                    (eval_users.__len__(), eval_dataframe.__len__()))

        logger.info("Eval data loaded")
        self.load = True
        self.split = False

    def make_train_test_split(self, slicing_interval=5, data_split=4):
        if self.load:

            self.training_frame = pd.DataFrame(columns=["user", "item", "rating", "timestamp"])
            self.testing_frame = pd.DataFrame(columns=["user", "item", "rating", "timestamp"])

            quantile = data_split * (1 / float(slicing_interval))
            logger.info("training on users dataset divides on quantile %s" % quantile)

            # TODO: later compare recommender results with using dataset having only data newer than from 2013

            for user in self.eval_users:
                user_reviews = self.eval_dataframe[self.eval_dataframe['user'] == user]

                # value dividing reviews of a user to training and testing
                slicing_timestamp = user_reviews["timestamp"].quantile(q=quantile)

                training_user_data = user_reviews[user_reviews["timestamp"] < slicing_timestamp]
                self.training_frame = self.training_frame.append(training_user_data)

                testing_user_data = user_reviews[user_reviews["timestamp"] >= slicing_timestamp]
                self.testing_frame = self.testing_frame.append(testing_user_data)

            self.split = True

        else:
            logger.info("no data yet loaded for make_train_test_split")

    def load_train_test_split(self, train_filepath, test_filepath):
        with open(train_filepath, "r") as train_file:
            self.training_frame = cPickle.load(train_file)
            logger.info("Loaded train set having %s reviews" % self.training_frame.__len__())

        with open(test_filepath, "r") as test_file:
            self.testing_frame = cPickle.load(test_file)
            logger.info("Loaded test set having %s reviews" % self.testing_frame.__len__())

        self.eval_users = self.training_frame["user"].unique()

        logger.info("Dataset contains reviews of %s users" % self.eval_users.__len__())

        self.split = True

    POSITIVE_RATING_THRESHOLD = 4

    def precision(self, y_true, y_pred):
        true_positives = 0

        true_good_ratings = filter(lambda rating: rating >= self.POSITIVE_RATING_THRESHOLD, y_true).__len__()
        for i in range(len(y_true)):
            true_positives += 1 if y_true[i] >= self.POSITIVE_RATING_THRESHOLD and \
                                 y_pred[i] >= self.POSITIVE_RATING_THRESHOLD else 0

        return float(true_positives)/true_good_ratings

    def recall(self, y_true, y_pred):
        true_positives = 0

        pred_good_ratings = filter(lambda rating: rating >= self.POSITIVE_RATING_THRESHOLD, y_pred).__len__()

        for i in range(len(y_true)):
            true_positives += 1 if y_true[i] >= self.POSITIVE_RATING_THRESHOLD and \
                                   y_pred[i] >= self.POSITIVE_RATING_THRESHOLD else 0

        return float(true_positives) / pred_good_ratings

    def f_score(self, y_true, y_pred, f=1):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return (1+(f**2))*(float(precision*recall)/((f**2*precision)+recall))

    def predict(self, tested_recommender):
        logger.info("Starting testing - gathering recommender predictions")

        self.y_true = list()
        self.y_pred = list()

        len_diff = 0
        delta_sum = 0
        for index, entry in self.testing_frame.iterrows():
            expected_score = entry["rating"]
            actual_score = tested_recommender.predict(entry["user"], entry["item"])

            # TODO: comment/uncomment for no output of matching
            # logger.info("expected - actual: %s - %s" % (expected_score, actual_score))

            if actual_score is not None:
                delta_sum += math.fabs(expected_score - actual_score)
                self.y_true.append(expected_score)
                self.y_pred.append(actual_score)
            else:
                len_diff += 1

        logger.info("Recommender method has predicted %s ratings" % (self.testing_frame.__len__() - len_diff))
        logger.info('Recommender failed to predict %s ratings', len_diff)

        return self.get_recent_results()

    # main method for simple evaluation
    # expects initialized Recommender class containing fit and predict methods
    # evaluates the results of the Recommender using the selected method
    # implemeted methods: MAE, RMSE, PRECISION, RECALL, F1
    def evaluate(self, recommender, pickled_train_filepath=None, pickled_test_filepath=None, all_data_filepath=None):
        """
        main method for simple evaluation
        expects initialized Recommender class containing fit and predict methods
        evaluates the results of the Recommender using the selected method
        implemented methods: MAE, RMSE, PRECISION, RECALL, F1
        """

        logger.info("Testing recommender implementation of %s " % recommender.__module__)
        logger.info("Using data from %s" % all_data_filepath if all_data_filepath else pickled_train_filepath)
        if all_data_filepath:
            if not self.load:
                logger.error("No data loaded. If no pickle boost is used, first use load_data(data_filepath) "
                             "to load data without pickling and run evaluation again.")
                return None
            if self.load and not self.split:
                logger.info("Computing train/test split on data %s" % all_data_filepath)
                self.make_train_test_split()
                logger.info("Train and test data split done")

        else:
            self.load_train_test_split(pickled_train_filepath, pickled_test_filepath)
            logger.info("Loaded pickled train and test data frames")

        logger.info("Starting recommender fit")
        recommender.fit(self.training_frame)
        logger.info("Recommender has fit")
        score = self.predict(recommender)
        logger.info("Recommender scores: %s" % score)
        return score

    def get_recent_results(self, method=None):
        """Gets the results of the latest evaluation as evaluated by selected method"""
        results = {"mae": mean_absolute_error(self.y_true, self.y_pred),
                  "rmse": np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
                  "precision": self.precision(self.y_true, self.y_pred),
                  "recall": self.recall(self.y_true, self.y_pred),
                  "f_score": self.f_score(self.y_true, self.y_pred)}
        if not method:
            return results
        else:
            return results[method]

    def pickle_split_dump(self, all_data_filepath, output_file_prefix, rating_range=(50, 50)):
        """Method to make a dump of the new split on data"""

        logger.info("Started pickling of train/test split on data from %s" % all_data_filepath)

        self.load_data(data_filepath=all_data_filepath, min_ratings=rating_range[0], user_is_robot_threshold=rating_range[1])

        logger.info("Done loading. Going to split the data.")
        logger.warn("This might take very long. Takes circa 30 mins on 100k data entries")

        self.make_train_test_split()

        logger.info("Split done. Writing outputs")

        pickle_file = "%s_%s_%s_%s.dat" % (output_file_prefix, rating_range[0], rating_range[1], "%s")

        with open(pickle_file % "train", "w") as pickle_file_writer:
            cPickle.dump(self.training_frame, pickle_file_writer)

        logger.info("Serialized %s training entries to a file %s " % (self.training_frame.__len__(), pickle_file % "train"))

        with open(pickle_file % "test", "w") as pickle_file_writer:
            cPickle.dump(self.testing_frame, pickle_file_writer)

        logger.info("Serialized %s testing entries to a file %s " % (self.testing_frame.__len__(), pickle_file % "test"))
