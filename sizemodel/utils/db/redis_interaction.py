import redis
import logging
import json
import os

log = logging.getLogger(__name__)

#TODO: Is this the best way to do this ?
NEW_CUSTOMER_PREFIX = 'new_customer'
EXISTING_CUSTOMER_PREFIX = 'existing_customer'
ARTICLE_PREFIX = 'article'

TIMESTAMP_SUFFIX = 'modelTimestamp'


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

    def set_existing_customer_df_sizes(self, customer_size_df):
        '''
        Store the customer sizes all at once rather than one at a time

        Args:
            customer_size_df: The dataframe with customer_id and size_object
        '''
        log.debug('Converting dataframe to dict for storage')
        size_dict = self._size_df_to_dict(
            size_df=customer_size_df,
            id_col='customer_id',
            prefix=EXISTING_CUSTOMER_PREFIX
        )
        log.debug('Storing dict')
        self.db.mset(size_dict)

    def set_new_customer_df_sizes(self, new_customer_size_df, model_timestamp=None):
        '''
        Store the new customer sizes all at once rather than one at a time

        Args:
            new_customer_size_df: The dataframe with id and size_object
            model_timestamp (timestamp): Whether to store the model_timestamp
        '''
        log.debug('Converting dataframe to dict for storage')
        size_dict = self._size_df_to_dict(
            size_df=new_customer_size_df,
            id_col='id',
            prefix=NEW_CUSTOMER_PREFIX
        )
        if model_timestamp is not None:
            ts_key = '{0}:{1}'.format(NEW_CUSTOMER_PREFIX, TIMESTAMP_SUFFIX)
            size_dict[ts_key] = str(model_timestamp)

        log.debug('Storing dict')
        self.db.mset(size_dict)

    def set_article_df_sizes(self, article_size_df):
        '''
        Store the article sizes all at once rather than one at a time

        Args:
            article_size_df: The dataframe with item_size_id and size_object

        '''
        log.debug('Converting dataframe to dict for storage')
        size_dict = self._size_df_to_dict(
            size_df=article_size_df,
            id_col='article_id',
            prefix=ARTICLE_PREFIX
        )
        log.debug('Storing dict')
        self.db.mset(size_dict)

    def get_single_existing_customer_sizes(self, customer_id):
        '''
        Get a given customers sizes

        Args:
            customer_id(int): The customers ID

        Returns:
            dict: The customers sizes
        '''

        key = '{0}:{1}'.format(EXISTING_CUSTOMER_PREFIX, str(customer_id))
        log.debug('Getting value from key: %s', key)

        str_sizes = self.db.get(key)
        import ipdb; ipdb.set_trace()
        return json.loads(str_sizes)

    def get_single_article_sizes(self, article_id):
        '''
        Get a given article sizes

        Args:
            article_id(int): The article ID

        Returns:
            dict: The articles sizes
        '''

        key = '{0}:{1}'.format(ARTICLE_PREFIX, str(article_id))
        log.debug('Getting value from key: %s', key)

        str_sizes = self.db.get(key)
        return json.loads(str_sizes)

    def get_single_new_customer_size(self, size_dimension, size):
        '''
        Get a single value for a single size dimention for a new customer.
        e.g What are the values for ShoeSize = 43.
        Please note that these sizes do not come with a value for sigma as we only have the single customer entered datapoint

        Args:
            size_dimension (string): What measure of size we are getting for (i.e shoeSize / trousersSizeWidth ... etc)
            size (string): The customer entered size (i.e 3XL or M ... etc)

        Returns:
            dict: A dictonary with the required sizes. Sample:
                    {
                        "name": 'shoeSize',
                        "mu": 13.2,
                        "sigma": None
                    }
        '''

        key = '{0}:{1}:{2}'.format(NEW_CUSTOMER_PREFIX, size_dimension, size)
        log.debug('Getting value from key: %s', key)

        str_sizes = self.db.get(key)
        try:
            return json.loads(str_sizes)
        except:
            import ipdb; ipdb.set_trace()

    def get_new_customer_size_model_timestamp(self):
        '''
        Get the modelTimestamp from the DB

        Returns:
            timestamp: The timestamp that the new customer models were created
        '''

        key = '{0}:{1}'.format(NEW_CUSTOMER_PREFIX, TIMESTAMP_SUFFIX)
        return str(self.db.get(key))

    def exists_in_db(self, id, prefix=EXISTING_CUSTOMER_PREFIX):
        '''
        Check whether or not a given article or customer has sizes associated with it

        Args:
            id: The article or customer ID
            prefix: The prefix used for storage

        Returns:
            bool: Whether or not the customer or article exists in the database
        '''

        key = '{0}:{1}'.format(prefix, str(id))
        return self.db.exists(key)

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
            '{0}:{1}'.format(prefix, str(row[id_col])) : row['size_object']
            for idx, row in size_df.iterrows()
        }

    # def set_single_existing_customer_sizes(self, customer_id, customer_sizes):
    #     '''
    #     Store a given customers sizes
    #
    #     Args:
    #         customer_id(int): The customers ID
    #         customer_sizes(dict): The customers sizes
    #     '''
    #     log.debug('Customer:Storing %s for %s', str(customer_sizes), customer_id)
    #
    #     key = 'customer:{0}'.format(str(customer_id))
    #     json_sizes = json.dumps(customer_sizes) #Redis can't store dictionaries directly
    #     self.db.set(key, json_sizes)
    #
    # def set_single_article_sizes(self, article_id, article_sizes):
    #     '''
    #     Store a given articles sizes
    #
    #     Args:
    #         article_id(int): The article ID
    #         article_sizes(dict): The articles sizes
    #     '''
    #     log.debug('Article: Storing %s for %s', str(article_sizes), article_id)
    #
    #     key = 'article:{0}'.format(str(article_id))
    #     json_sizes = json.dumps(article_sizes) #Redis can't store dictionaries directly
    #     self.db.set(key, json_sizes)



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