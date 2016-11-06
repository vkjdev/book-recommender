class MeanRatingRecommender:

    mean = 0

    def __init__(self):
        pass

    # expects raw entries in format "user,item,rating,timestamp"
    def fit(self, raw_data):
        self.mean = raw_data["rating"].mean()
        return True

    # expects one entry's user and item
    # will return predicted rating which is the only attribute to be used for performance evaluation
    def predict(self, user, item):

        return self.mean
