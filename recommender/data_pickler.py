import logging
import pandas as pd
import cPickle
import numpy as np

DATA_FILE_PATH="/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/ratings_Books.csv"
# DATA_FILE_PATH = '/home/kvassay/data/book-recommender/ratings_Books.csv'
USERS_TO_COLLECT = 80
SOFT_USERS_TO_COLLECT = 50

SAMPLED_USERS_CONTENT = [30, 50]

PICKLE_FILE_DATA = "%s_users_ratings_data.dat" % USERS_TO_COLLECT
PICKLE_FILE_TRAIN = "%s_users_ratings_train.dat" % USERS_TO_COLLECT
PICKLE_FILE_TEST = "%s_users_ratings_test.dat" % USERS_TO_COLLECT

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

# slices to use for testing methods improvements on increasing amount of testing data
SLICING_INTERVAL = 5

# select how many times the evaluation will split data and test
# selecting 1 means one split with fold on SLICING_INTERVAL-1/SLICING_INTERVAL timestamp for every user
# can automatically test a development of model performance on increasing amount of training data
SLICING_RUNS = 1


def train_test_split(eval_run, slicing_interval):
    training_frame = pd.DataFrame(columns=["user", "item", "rating", "timestamp"])
    testing_frame = pd.DataFrame(columns=["user", "item", "rating", "timestamp"])

    quantile = eval_run * (1 / float(slicing_interval))
    logger.info("training/testing split on users dataset divides on quantile %s" % quantile)

    for user in content_to_pickle["user"]:
        user_reviews = content_to_pickle[content_to_pickle['user'] == user]

        # value dividing reviews of a user to training and testing
        slicing_timestamp = user_reviews["timestamp"].quantile(q=quantile)

        training_user_data = user_reviews[user_reviews["timestamp"] < slicing_timestamp]
        training_frame = training_frame.append(training_user_data)

        testing_user_data = user_reviews[user_reviews["timestamp"] >= slicing_timestamp]
        testing_frame = testing_frame.append(testing_user_data)

    logger.info("training dataframe size: %s" % training_frame.__len__())
    logger.info("testing dataframe size: %s" % testing_frame.__len__())

    return training_frame, testing_frame

logger.info("File %s reading" % data_file)
with open(data_file, "r") as f:
    df = pd.read_csv(data_file)
logger.info("File loaded")

grouped_users = df.groupby(["user"]).count()
bound_users = grouped_users[grouped_users["item"] <= SAMPLED_USERS_CONTENT[1]]
bound_users = bound_users[bound_users["item"] >= SAMPLED_USERS_CONTENT[0]]

bound_users_content = df[df["user"].isin(bound_users.index.values)]
# good init user: A3MH40TK0FRBYG, A3EVHLQTSVFI49
# init_user = np.random.choice(bound_users.index.values)
init_user = "A3EVHLQTSVFI49"
logger.info("Init user: %s" % init_user)

CONTENT_INTERSECT_THRESHOLD = 10

seen_content = set(bound_users_content[bound_users_content["user"] == init_user]["item"])
seen_content_data = bound_users_content[bound_users_content["item"].isin(seen_content)]

pickled_users = set()
pickled_users.add(init_user)
addition = 1
users_alpha = 0
while True:

    if addition == 0:
        logger.warn("DEADLOCK for init user %s. Traversed %s ratings of %s users" % (init_user,
                                                                                     seen_content_data.__len__(),
                                                                                     pickled_users.__len__()))
        init_user = np.random.choice(bound_users.index.values)
        logger.info("selecting new init user %s" % init_user)
        # pickled_users.add(init_user)
        new_seen_content = set(bound_users_content[bound_users_content["user"] == init_user]["item"])

        users_delta = len(pickled_users) - users_alpha
        if users_delta >= SOFT_USERS_TO_COLLECT:
            logger.warn("%s users and their ratings left in collected dataset" % len(pickled_users))
            seen_content_data.append(bound_users_content[bound_users_content["item"].isin(new_seen_content)])
        else:
            logger.warn("Browse restart")
            seen_content_data = bound_users_content[bound_users_content["item"].isin(new_seen_content)]
            # seen_content_data = bound_users_content[bound_users_content["item"].isin(seen_content)]
            # pickled_users = set()
        seen_content.update(new_seen_content)
        users_alpha = len(pickled_users)


    addition = 0
    for user in set(seen_content_data["user"].unique()):
        no_content_increase = 0
        # if user not in bound_users_contents:
        #     continue
        intersect_content = seen_content_data[seen_content_data["user"] == user]
        intersect_content = intersect_content[intersect_content["item"].isin(seen_content)]
        logger.debug("User %s have currently %s ratings" % (user, len(intersect_content)))
        if intersect_content.__len__() >= CONTENT_INTERSECT_THRESHOLD:
            logger.info("Found user %s intersecting with %s ratings" % (user, intersect_content.__len__()))
            new_user_data = bound_users_content[bound_users_content["user"] == user]
            logger.info("adding %s ratings to selected dataset" % new_user_data.__len__())

            seen_content_data.append(new_user_data)
            seen_content.update(intersect_content["item"])

            if user not in pickled_users:
                addition += 1
            pickled_users.add(user)

            if len(pickled_users) >= USERS_TO_COLLECT:
                break
            logger.info("STATE INFO: %s users with %s ratings" % (pickled_users.__len__(), seen_content_data.__len__()))

    if len(pickled_users) >= USERS_TO_COLLECT:
        break

content_to_pickle = bound_users_content[bound_users_content["user"].isin(pickled_users)]
content_to_pickle = content_to_pickle[content_to_pickle["item"].isin(seen_content)]

# dataset test:
gbu = content_to_pickle.groupby(["user"]).count()["item"]
gbi = content_to_pickle.groupby(["item"]).count()["user"]
logger.info("SELECTED DATASET:")
logger.info("Selected %s users" % len(pickled_users))
logger.info("Selected users content have %s ratings" % len(content_to_pickle))

logger.info("Dataset density:")
logger.info("Average no of ratings for user: %s " % gbu.mean())
logger.info("Average no of ratings for item: %s " % gbi.mean())

print(pickled_users)

for eval_run in range(SLICING_INTERVAL - SLICING_RUNS, SLICING_INTERVAL):
    train_data, test_data = train_test_split(eval_run, SLICING_INTERVAL)

logger.info("Serializing %s data entries to a file %s " % (content_to_pickle.__len__(), PICKLE_FILE_DATA))
with open(PICKLE_FILE_TRAIN, "w") as pickle_file_writer:
    cPickle.dump(train_data, pickle_file_writer)

logger.info("Serialized %s training entries to a file %s " % (train_data.__len__(), PICKLE_FILE_TRAIN))

with open(PICKLE_FILE_TEST, "w") as pickle_file_writer:
    cPickle.dump(test_data, pickle_file_writer)

logger.info("Serialized %s testing entries to a file %s " % (test_data.__len__(), PICKLE_FILE_TEST))
