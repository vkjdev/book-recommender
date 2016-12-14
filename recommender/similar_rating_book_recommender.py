import math
from pandas import DataFrame
import numpy as np
import logging
import scipy
from scipy.stats.stats import pearsonr as correlation_coef


logger = logging.getLogger()
logger.setLevel(20)

FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

date_format = "%Y-%m-%d %H:%M:%S"
np.set_printoptions(threshold=np.nan)
class SimilarRatingBookRecommender:

    user_mean = dict()
    item_mean = dict()

    sorted_user_vector = list()
    sorted_book_vector = list()

    result_array = np.array([])
    user_item_matrix = None

    distance_model = None
    mean = 0

    def __init__(self):
        self.transposed_matrix = None

    # expects raw entries in format "user,item,rating,timestamp"
    def fit(self, raw_data):
        # build a normalized vector of users ratings for each book and train a knn model on it
        logger.info("Selecting and sorting users and items")

        self.sorted_user_vector = np.sort(raw_data["user"].unique())
        self.sorted_book_vector = np.sort(raw_data["item"].unique())
        # TODO: normalize on predict if necessary
        for item in self.sorted_book_vector:
             self.item_mean[item] = raw_data[raw_data["item"] == item]["rating"].mean()

        logger.info("%s items processed" % self.sorted_book_vector.__len__())

        # self.user_item_matrix = np.zeros(shape=(len(self.sorted_user_vector), len(self.sorted_book_vector)))
        self.user_item_matrix = scipy.sparse.lil_matrix(
            (len(self.sorted_user_vector), len(self.sorted_book_vector)))

        user_iter = 0
        for user in self.sorted_user_vector:
            user_entries = raw_data[raw_data["user"] == user]
            self.user_mean[user] = user_entries["rating"].mean()
            # user_entries["rating"] = _normalize_user_ratings(user_entries["rating"])

            # user_sparse_items = map(lambda book: user_entries[user_entries["item"] == book]["rating"]
            # if book in user_items else 0, self.sorted_book_vector)
            user_sparse_reviews = (np.zeros(shape=len(self.sorted_book_vector)))
            user_sparse_reviews[self.sorted_book_vector.searchsorted(user_entries["item"])] = user_entries["rating"]

            # user_sparse_items = np.array(user_sparse_items)
            self.user_item_matrix[user_iter, :] = user_sparse_reviews

            user_iter += 1

            logger.info("%s users processed" % user_iter)

            # print training_struct.describe()
        print self.user_item_matrix
        # self.distance_model.fit(self.user_item_matrix)

        return True

#TODO make comments
    def predict(self, user, item):
        if user in self.sorted_user_vector:

            user_index = self.sorted_user_vector.tolist().index(user)
            user_ratings = self.user_item_matrix[user_index]

            if item in self.sorted_book_vector:

                self.transposed_matrix = np.transpose(self.user_item_matrix)
                item_index = self.sorted_book_vector.searchsorted(item)
                item_ratings = self.transposed_matrix[item_index]
                user_indices = np.nonzero((self.transposed_matrix[item_index])[0])[1]
                rating_index = 0
                for item in user_indices:
                    similar_user_ratings = self.user_item_matrix[item]
                    refference_rating = item_ratings.data[0][rating_index]
                    result = filter(lambda x : x <= refference_rating + 1 and x >= refference_rating - 1, similar_user_ratings.data[0])
                    self.result_array = np.append(self.result_array, result)

                result = math.ceil(np.mean(self.result_array))
                var = math.ceil(np.mean(self.transposed_matrix[item_index].data[0]))
                return result
            else:
                user_mean = self.user_mean[user]

                return math.ceil(user_mean)
        else:
            if item in self.sorted_book_vector:
                self.transposed_matrix = np.transpose(self.user_item_matrix)
                item_index = self.sorted_book_vector.searchsorted(item)
                print self.transposed_matrix[item_index].data[0]
                return math.ceil(np.mean(self.transposed_matrix[item_index].data[0]))
            else:
                return math.ceil(np.mean(self.item_mean.values()))

