import redis
import logging
import pickle as pkl
import json
import os

log = logging.getLogger(__name__)

class RedisStore():
    '''
    Helper class to deal with the interactions with the redis DB
    '''

    def __init__(self, host=os.environ['DB_REDIS_ADDR'], port=os.environ['DB_REDIS_PORT']):
        '''
        Initialize the class

        Args:
            host: The host address of the dataRedibase
            port: The port of the database
        '''

        self.db = redis.Redis(
            host=host,
            port=port
        )

        log.info('Initalised the redis database with host: %s and port: %s', host, port)

    def set_single_customer_sizes(self, customer_id, customer_sizes):
        '''
        Store a given customers sizes

        Args:
            customer_id(int): The customers ID
            customer_sizes(dict): The customers sizes
        '''
        log.debug('Customer:Storing %s for %s', str(customer_sizes), customer_id)

        key = 'customer:{0}'.format(str(customer_id))
        json_sizes = json.dumps(customer_sizes) #Redis can't store dictionaries directly
        self.db.set(key, pkl_sizes)

    def set_customer_df_sizes(self, customer_size_df):
        '''
        Store the customer sizes all at once rather than one at a time

        Args:
            customer_size_df: The dataframe with customer_id and size_object
        '''
        log.info('Converting dataframe to dict for storage')
        size_dict = self._size_df_to_dict(
            size_df=customer_size_df,
            id_col='customer_id',
            prefix='customer'
        )
        log.info('Storing dict')
        self.db.mset(size_dict)

    def set_article_df_sizes(self, article_size_df):
        '''
        Store the article sizes all at once rather than one at a time

        Args:
            article_size_df: The dataframe with item_size_id and size_object
        '''
        log.info('Converting dataframe to dict for storage')
        size_dict = self._size_df_to_dict(
            size_df=article_size_df,
            id_col='item_size_id',
            prefix='article'
        )
        log.info('Storing dict')
        self.db.mset(size_dict)


    def get_single_customer_sizes(self, customer_id):
        '''
        Get a given customers sizes

        Args:
            customer_id(int): The customers ID

        Returns:
            dict: The customers sizes
        '''
        log.debug('Retrieving sizes for customer %s', customer_id)

        key = 'customer:{0}'.format(str(customer_id))
        str_sizes = self.db.get(key)
        return json.loads(str_sizes)


    def set_single_article_sizes(self, article_id, article_sizes):
        '''
        Store a given articles sizes

        Args:
            article_id(int): The article ID
            article_sizes(dict): The articles sizes
        '''
        log.debug('Article: Storing %s for %s', str(article_sizes), article_id)

        key = 'article:{0}'.format(str(article_id))
        pkl_sizes = pkl.dumps(article_sizes) #Redis can't store dictionaries directly
        self.db.set(key, pkl_sizes)


    def get_article_sizes(self, article_id):
        '''
        Get a given article sizes

        Args:
            article_id(int): The article ID

        Returns:
            dict: The articles sizes
        '''
        log.debug('Retrieving article sizes for %s', article_id)

        key = 'article:{0}'.format(str(article_id))
        str_sizes = self.db.get(key)
        return pkl.loads(str_sizes)


    def _size_df_to_dict(self, size_df, id_col, prefix):
        '''
        Convert a size_df to a dictionary for storage

        Args:
            id_col: The column to use for keys
            prefix: What to prefix these keys with

        Returns:
            dict: dictionary with key,value in correct format for storage
        '''

        return {
            '{0}:{1}'.format(prefix, str(row[id_col])) : json.dumps(row['size_object'])
            for idx, row in size_df.iterrows()
        }



# if __name__ == '__main__':
#     redisDB = RedisStore()
    # redisDB.set_customer_sizes(-1, {'size': 'test_customer'})
    # print(redisDB.get_customer_sizes(-1))
    # redisDB.set_article_sizes(-1, {'size': 'test_article'})
    # print(redisDB.get_article_sizes(-1))

    # import pandas as pd
    # df = pd.DataFrame({
    #     'customer_id': [1,2,3],
    #     'size_object': [{'size': 'test_customer1'}, {'size': 'test_customer2'}, {'size': 'test_customer3'}]
    # })
    # redisDB.set_customer_df_sizes(df)