from pandas import DataFrame
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats.stats import pearsonr as correlation_coef
import math


def _normalize_user_ratings(user_ratings):
    rating_quantiles = map(lambda rating: float((np.sum(user_ratings.le(rating)))/len(user_ratings)*100)-50, user_ratings._values)
    return rating_quantiles


def _recall(searched_items, retrieved_items):
    intersection_size = len(set(searched_items) & set(retrieved_items))
    return float(intersection_size) / len(searched_items)


class KnnRecommender:

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
        training_struct = DataFrame(columns=["book_id"])

        self.sorted_user_vector = np.sort(raw_data["user"].unique())
        self.sorted_book_vector = np.sort(raw_data["item"].unique())

        for item in self.sorted_book_vector:
            self.item_mean[item] = raw_data[raw_data["item"] == item]["rating"].mean()

        self.user_item_matrix = np.zeros(shape=(len(self.sorted_user_vector), len(self.sorted_book_vector)))

        user_iter = 0
        for user in self.sorted_user_vector:
            user_entries = raw_data[raw_data["user"] == user]
            self.user_mean[user] = user_entries["rating"].mean()
            user_entries["norm_rating"] = _normalize_user_ratings(user_entries["rating"])

            user_items = np.sort(user_entries["item"].unique())
            user_sparse_items = map(lambda book: user_entries[user_entries["item"] == book]["norm_rating"] if book in user_items else 0, self.sorted_book_vector)
            self.user_item_matrix[user_iter] = user_sparse_items

            user_iter += 1

        # print training_struct.describe()
        print self.user_item_matrix
        self.distance_model.fit(self.user_item_matrix)

        return True

    # expects one entry's user and item
    # will return predicted rating which is the only attribute to be used for performance evaluation
    def predict(self, user, item):
        # tries to match user's positively rated items with other users
        # and try to find a predicted book in ratings of the similar users

        if user in self.sorted_user_vector:

            user_index = self.sorted_user_vector.tolist().index(user)
            user_ratings = self.user_item_matrix[user_index]
            user_positive_ratings = np.where(user_ratings > 0)[0]

            if item in self.sorted_book_vector:
                # main thread
                # both user and item seen on previous data
                # similar items = items with high correlation, rating wise
                item_user_matrix = np.transpose(self.user_item_matrix)

                item_data = item_user_matrix[self.sorted_book_vector.tolist().index(item)]
                # item_similarity = np.vectorize(lambda row: correlation_coef(row, item_data)[0])(item_user_matrix)
                item_similarity = map(lambda row: correlation_coef(row, item_data)[0], item_user_matrix)
                correlated_percentile = np.percentile(item_similarity, q=80)
                correlated_items = np.where(item_similarity >= correlated_percentile)[0]

                norm_result = (float(self.user_mean[user])/2) + \
                              (_recall(user_positive_ratings, correlated_items)*self.user_mean[user]/2)
                norm_result = math.floor(norm_result)

            else:
                # no item seen before
                # compute user's mean rating and return that
                user_mean = self.user_mean[user]

                norm_result = math.ceil(user_mean)
        else:
            if item in self.sorted_book_vector:
                # no user seen before

                norm_result = math.ceil(self.item_mean[item])
            else:
                # no information previously seen - use mean of all users
                norm_result = math.ceil(np.mean(self.item_mean.values()))

        return norm_result

