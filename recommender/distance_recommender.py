import scipy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import logging
import math

logger = logging.getLogger()


def _normalize_user_ratings(user_ratings):
    norm_values = scipy.stats.zscore(user_ratings, axis=None)
    if np.isnan(norm_values).any():
        # all zscores are not defined, if all ratings in the set are the same, thus have sero variance
        norm_values = np.array([1.5*user_ratings.mean()-5]*max(user_ratings.shape))
    if norm_values.shape.__len__() > 2:
        print norm_values
    return norm_values


def _denormalize_rating(ratings, norm_rating):
    std = np.std(ratings)
    mean = np.mean(ratings)

    # return np.apply_along_axis(lambda z: z*std + mean, axis=0, arr=norm_ratings)
    return norm_rating*std + mean


def _recall(searched_items, retrieved_items):
    intersection_size = len(set(searched_items) & set(retrieved_items))
    return float(intersection_size) / len(searched_items)


class UserDistanceRecommender:

    user_mean = dict()
    item_mean = dict()

    sorted_user_vector = list()
    sorted_book_vector = list()
    user_item_matrix = None

    distance_model = NearestNeighbors(n_jobs=-1)

    def __init__(self):
        pass

    # expects raw entries in format "user,item,rating,timestamp"
    def fit(self, raw_data):
        # build a normalized vector of users ratings for each book and train a knn model on it
        logger.info("Selecting and sorting users and items")

        unique_users = raw_data["user"].unique()

        self.sorted_user_vector = np.sort(unique_users)
        self.sorted_book_vector = np.sort(raw_data["item"].unique())

        # self.user_item_matrix = np.zeros(shape=(len(self.sorted_user_vector), len(self.sorted_book_vector)))
        self.user_item_matrix = scipy.sparse.csr_matrix((len(self.sorted_user_vector), len(self.sorted_book_vector)))

        user_iter = 0
        for user in self.sorted_user_vector:
            user_entries = raw_data[raw_data["user"] == user]
            self.user_mean[user] = user_entries["rating"].mean()

            # user_sparse_items = map(lambda book: user_entries[user_entries["item"] == book]["rating"]
            # if book in user_items else 0, self.sorted_book_vector)
            user_sparse_reviews = (np.zeros(shape=len(self.sorted_book_vector)))
            user_sparse_reviews[self.sorted_book_vector.searchsorted(user_entries["item"])] = user_entries["rating"]

            # user_sparse_items = np.array(user_sparse_items)
            self.user_item_matrix[user_iter, :] = user_sparse_reviews

            user_iter += 1

            # logger.info("%s/%s users processed" % (user_iter, len(unique_users)))

        # print training_struct.describe()
        self.distance_model.fit(self.user_item_matrix)

        return True

    # expects one entry's user and item
    # will return predicted rating which is the only attribute to be used for performance evaluation
    def predict(self, user, item):
        try:
            item_index = self.sorted_book_vector.tolist().index(item)
        except KeyError:
            try:
                user_index = self.sorted_user_vector.tolist().index(user)
                user_mean = self.user_item_matrix[user_index].mean()
                return math.floor(user_mean)

            except KeyError:
                # no item, no user seen before
                # returns overall mean
                overall_mean = self.user_item_matrix[np.nonzero(self.user_item_matrix)].mean()
                return math.floor(overall_mean)

        try:
            user_index = self.sorted_user_vector.tolist().index(user)
        except KeyError:
            try:
                item_index = self.sorted_book_vector.tolist().index(item)
                item_mean = self.user_item_matrix[:, item_index].mean()
                return math.floor(item_mean)

            except KeyError:
                # no item, no user seen before
                # returns overall mean
                overall_mean = self.user_item_matrix[np.nonzero(self.user_item_matrix)].mean()
                return math.floor(overall_mean)

        user_rating_vector = self.user_item_matrix[user_index]
        user_rating_vector[np.nonzero(user_rating_vector)] = _normalize_user_ratings(user_rating_vector[np.nonzero(user_rating_vector)])
        transposed_matrix = self.user_item_matrix.transpose()
        # find users that rated the current book and predict the rating of the closest as in euclidean space
        users_rated_item_ratings_indices = np.nonzero(transposed_matrix[item_index][0].data)[0]

        current_min_distance = np.inf
        current_best_vector = None
        for matched_user_ratings_index in users_rated_item_ratings_indices:
            rated_user_vector = self.user_item_matrix[matched_user_ratings_index]
            rated_user_vector[np.nonzero(rated_user_vector)] = _normalize_user_ratings(rated_user_vector[np.nonzero(rated_user_vector)])
            distance_from_current_user = euclidean_distances(user_rating_vector, rated_user_vector)
            if distance_from_current_user <= current_min_distance:
                current_min_distance = distance_from_current_user
                current_best_vector = rated_user_vector
                current_best_index = matched_user_ratings_index

        logger.debug("Most similar user distance: %s" % current_min_distance)

        # best matching user has now the highest indices
        if current_min_distance < np.inf:
            # consider the closest user as close enough and return the exact match
            # de-normalize rating by current user's normalization
            denorm_best_rating = _denormalize_rating(self.user_item_matrix[:,
                                                     np.nonzero(self.user_item_matrix[user_index, :])[1]].data,
                                                     current_best_vector[0, item_index])
            logger.debug("Normalized rating: %s" % denorm_best_rating)
            return math.floor(denorm_best_rating) if math.floor(denorm_best_rating) <= 5 else 5
        else:
            logger.warn("User %s not seen before" % user)

            return self.user_item_matrix[np.nonzero(self.user_item_matrix[:, item_index])[0], item_index].mean()

