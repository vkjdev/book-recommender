import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import logging
import time

logger = logging.getLogger()


class Doc2VecRecommender():
    def __init__(self):
        logger.info('Initializing Doc2Vec Recommender...')
        self.model = None
        self.user_items_dict = None

    @staticmethod
    def get_user_items_dict(data, min_ratings, max_ratings):
        logger.info('Extracting User:[items] dict from dataframe.')
        user_item_dict = dict()
        cntr = 1
        start = time.time()
        for row in data.itertuples():
            if cntr % 100000 == 0:
                dur = time.time() - start
                start = time.time()
                logger.info('Transformed %s rows to user:items dict. Transforming last 100000 rows took %ss...', cntr,
                            dur)
            if row.user in user_item_dict:
                user_item_dict[row.user].append(row.item)
            else:
                user_item_dict[row.user] = [row.item]
            cntr += 1
        user_item_dict_clean = {key: value for key, value in user_item_dict.iteritems() if
                                len(value) >= min_ratings and len(value) < max_ratings}
        return user_item_dict_clean

    @staticmethod
    def get_user_items_list_lengths(user_item_dict):
        return pd.DataFrame(sorted([len(item_list) for _, item_list in user_item_dict.iteritems()]))

    def fit(self, raw_data):
        self.user_items_dict = Doc2VecRecommender.get_user_items_dict(raw_data, min_ratings=3, max_ratings=100)
        logger.info('Transforming training data rows to Tagged Documents for Gensim...')
        train_data = [TaggedDocument(words, [user_id]) for user_id, words in self.user_items_dict.iteritems()]
        logger.info('Training Doc2Vec model on %s examples.', len(train_data))
        start = time.time()
        self.model = Doc2Vec(train_data, dm=0, size=30, window=8, min_count=1, workers=4)
        dur = time.time() - start
        logger.info('Training finished with duration: %ss...', dur)

    def get_recommendations(self, user, item, k):
        similar_list = self.model.most_similar(item, topn=k)
        asin_list = [item[0] for item in similar_list]
        return asin_list

    @staticmethod
    def cosine_sim_to_rating(sim):
        # cosine sim gets values from -1 to 1, 1 meaning exactly the same
        if sim <= -0.6:
            return 1
        if sim <= -0.2:
            return 2
        if sim <= 0.2:
            return 3
        if sim <= 0.6:
            return 4
        return 5

    def get_rating(self, user, item):
        if user in self.user_items_dict:
            users_items = [item for item in self.user_items_dict[user] if item in self.model.vocab]
            if len(users_items) > 0:
                if item in self.model.vocab:
                    cos_sim = self.model.n_similarity(users_items, [item])
                    return Doc2VecRecommender.cosine_sim_to_rating(cos_sim)
                else:
                    return None
            else:
                return None
        else:
            return None

    def predict(self, user, item):
        return self.get_rating(user, item)
