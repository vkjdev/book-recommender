import pandas as pd
from recommender.util import get_cfg
from csv import DictReader, Dialect


class BooksDatasetReader():
    def __init__(self, cfg):
        self.cfg = get_cfg(cfg)
        data_folder = self.cfg['recommender']['dataset']['dataset_folder_path']
        self.users_path = data_folder + self.cfg['recommender']['dataset']['users_file_name']
        self.books_path = data_folder + self.cfg['recommender']['dataset']['books_file_name']
        self.ratings_path = data_folder + self.cfg['recommender']['dataset']['ratings_file_name']

    def read_from_csv(self, file_path, return_type='pandas_df', separator=';', quotechar='"'):
        """
        :param return_type: 'pandas_df' or 'dict'
        """
        if return_type == 'pandas_df':
            return pd.read_csv(file_path, sep=separator, quotechar=quotechar, quoting=1, error_bad_lines=False)
        else:
            dialect = Dialect()
            dialect.delimiter = separator
            dialect.quotechar = quotechar
            with open(file_path) as csvfile:
                reader = DictReader(csvfile, dialect=dialect)
            return reader

    def read_books(self, return_type='pandas_df'):
        """
        :param return_type: 'pandas_df' or 'dict'
        """
        return self.read_from_csv(self.books_path, return_type=return_type)

    def read_users(self, return_type='pandas_df'):
        """
        :param return_type: 'pandas_df' or 'dict'
        """
        return self.read_from_csv(self.users_path, return_type=return_type)

    def read_ratings(self, return_type='pandas_df'):
        """
        :param return_type: 'pandas_df' or 'dict'
        """
        return self.read_from_csv(self.ratings_path, return_type=return_type)

    def read_all(self, return_type='pandas_df'):
        """
        :param return_type: 'pandas_df' or 'dict'
        :return: tuple of return types in format: (users, books, ratings)
        """
        return self.read_users(return_type=return_type), self.read_books(return_type=return_type), self.read_ratings(
            return_type=return_type)
