import redis
import logging
import pickle as pkl

log = logging.getLogger(__name__)

class RedisStore():
    '''
    Helper class to deal with the interactions with the redis DB
    '''

    def __init__(self, host='production-sizeapi-redis-3.apps.outfittery.de', port=6379):
        '''
        Initialize the class

        Args:
            host: The host address of the database
            port: The port of the database
        '''

        self.db = redis.Redis(
            host=host,
            port=port
        )

        log.info('Initalised the redis database with host: %s and port: %s', host, port)

    def set_customer_sizes(self, customer_id, customer_sizes):
        '''
        Store a given customers sizes

        Args:
            customer_id(int): The customers ID
            customer_sizes(dict): The customers sizes
        '''
        log.debug('Customer:Storing %s for %s', str(customer_sizes), customer_id)

        key = 'customer:{0}'.format(str(customer_id))
        pkl_sizes = pkl.dumps(customer_sizes) #Redis can't store dictionaries directly
        self.db.set(key, pkl_sizes)


    def get_customer_sizes(self, customer_id):
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
        return pkl.loads(str_sizes)


    def set_article_sizes(self, article_id, article_sizes):
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



if __name__ == '__main__':
    redisDB = RedisStore()
    redisDB.set_customer_sizes(-1, {'size': 'test_customer'})
    print(redisDB.get_customer_sizes(-1))
    redisDB.set_article_sizes(-1, {'size': 'test_article'})
    print(redisDB.get_article_sizes(-1))