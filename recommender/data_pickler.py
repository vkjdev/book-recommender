import logging
import pandas as pd
import cPickle

DATA_FILE_PATH="/home/michal/Documents/Misc/recommenders/vcs/book-recommender/data/ratings_Books.csv"
# DATA_FILE_PATH = '/home/kvassay/data/book-recommender/ratings_Books.csv'

SAMPLED_USERS_CONTENT = [20, 50]

PICKLE_FILE_DATA = "%s_%s_ratings_data.dat" % (SAMPLED_USERS_CONTENT[0], SAMPLED_USERS_CONTENT[1])
PICKLE_FILE_TRAIN = "%s_%s_ratings_train.dat" % (SAMPLED_USERS_CONTENT[0], SAMPLED_USERS_CONTENT[1])
PICKLE_FILE_TEST = "%s_%s_ratings_test.dat" % (SAMPLED_USERS_CONTENT[0], SAMPLED_USERS_CONTENT[1])

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
    logger.info("training on users dataset divides on quantile %s" % quantile)

    for user in filtered_users:
        user_reviews = filtered_data[filtered_data['user'] == user]

        # value dividing reviews of a user to training and testing
        slicing_timestamp = user_reviews["timestamp"].quantile(q=quantile)

        training_user_data = user_reviews[user_reviews["timestamp"] < slicing_timestamp]
        training_frame = training_frame.append(training_user_data)

        testing_user_data = user_reviews[user_reviews["timestamp"] >= slicing_timestamp]
        testing_frame = testing_frame.append(testing_user_data)

    logger.info("training dataframe size: %s" % training_frame.__len__())
    logger.info("testing dataframe size: %s" % testing_frame.__len__())

    return training_frame, testing_frame

pickled_users = list()

logger.info("File %s reading" % data_file)
with open(data_file, "r") as f:
    df = pd.read_csv(data_file)
logger.info("File loaded")

grouped_users = df.groupby(["user"]).count()
filtered_users = grouped_users[grouped_users["item"] <= SAMPLED_USERS_CONTENT[1]]
filtered_users = filtered_users[filtered_users["item"] >= SAMPLED_USERS_CONTENT[0]].index.values

logger.info("Selected %s users" % len(filtered_users))

print(filtered_users)

filtered_data = df[df["user"].isin(filtered_users)]

logger.info("Selected %s filtered users entries" % len(filtered_data))

for eval_run in range(SLICING_INTERVAL - SLICING_RUNS, SLICING_INTERVAL):
    train_data, test_data = train_test_split(eval_run, SLICING_INTERVAL)

with open(PICKLE_FILE_DATA, "w") as pickle_file_writer:
    cPickle.dump(filtered_data, pickle_file_writer)

logger.info("Serialized %s data entries to a file %s " % (filtered_data.__len__(), PICKLE_FILE_DATA))

with open(PICKLE_FILE_TRAIN, "w") as pickle_file_writer:
    cPickle.dump(train_data, pickle_file_writer)

logger.info("Serialized %s training entries to a file %s " % (train_data.__len__(), PICKLE_FILE_TRAIN))

with open(PICKLE_FILE_TEST, "w") as pickle_file_writer:
    cPickle.dump(test_data, pickle_file_writer)

logger.info("Serialized %s testing entries to a file %s " % (test_data.__len__(), PICKLE_FILE_TEST))
