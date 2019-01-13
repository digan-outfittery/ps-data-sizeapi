import jsonschema
from jsonschema import Draft4Validator as Validator
import logging

from sizemodel.utils.db.redis_interaction import RedisStore
from sizemodel.app.deciders.decider_base import BaseDecider
from sizemodel.app.utils.json_utils import load_json_resource

SizeModelRequest = load_json_resource(__name__, 'resources/SizeModelRequest.json')
SizeModelResponse = load_json_resource(__name__, 'resources/SizeModelResponse.json')

log = logging.getLogger(__name__)


class ATLSizeModelDecider(BaseDecider):
    '''
    A decider that simply returns random items from current stock that the user has not given feedback on before
    '''

    def __init__(self):
        super().__init__('atl_sizemodel_decider')
        self.redis_store = RedisStore()

    def validate_request(self, request):
        jsonschema.validate(request, SizeModelRequest, cls=Validator)

    def validate_response(self, response):
        jsonschema.validate(response, SizeModelResponse, cls=Validator)

    def decide(self, request_data):
        '''

        Args:
            request_data (dict): The json request data as a dictionary

        Returns:
            dictionary: The response as a dictionary with the sizes of the customer as well as the articles
        '''

        response = {}
        response['data'] = request_data['data']
        is_new_customer = self._is_new_customer(request_data)

        #Get the customer sizes
        customer_sizes = self._get_new_customer_size(request_data['attributes']['customer']) if is_new_customer \
            else self._get_existing_customer_size(request_data['attributes']['customerId'])

        #Get the article sizes
        article_sizes = self._get_article_sizes(request_data['attributes']['articles'])

        all_sizes = {
            'customer': customer_sizes,
            'articles': article_sizes
        }

        response['attributes'] = all_sizes

        return response


    def _get_article_sizes(self, article_ids):
        '''
        Get the mean & std dev of the size of each of the articles

        Args:
            article_ids: A list of article ID's to get size info for

        Returns:
            list: A list of dictionaries. Each dictionary is an article Ids with it's size. Example:
                            {
                                "articleId": 122345_1224,
                                "size": {
                                    "mu": 12.4,
                                    "sigma": 3.2
                                }
                            }

        '''

        retlist = []
        for article in article_ids:
            size_object = self.redis_store.get_single_article_sizes(article)
            retlist.append(size_object)

        return retlist

    def _is_new_customer(self, request_data):
        '''
        Is a customer new or repeat

        Args:
            request_data: The request data sent to the API

        Returns:
            bool: Whether or not the given customer is a new customer
        '''

        customer_id = request_data['attributes']['customerId']
        return self.redis_store.exists_in_db(customer_id) == 0

    def _get_new_customer_size(self, customer_object):
        '''
        Get the mean & std dev of the size of a specific new customer

        Args:
            customer_object: The customer object to get sizes for

        Returns:
            dictionary: A Dictionary with the customer_id and that customers sizes. Example:
                            {
                                "customerId": 122345_1224,
                                "isFirstTimeCustomer": True,
                                "sizes": [
                                    {
                                        "name": "shoeSize",
                                        "mu": 12.4,
                                        "sigma": None
                                    },
                                    {
                                        "name": "tShirtSize",
                                        "mu": 12.4,
                                        "sigma": None
                                    },
                                    {
                                        "name": "trouserSizeWidth",
                                        "mu": 12.4,
                                        "sigma": None
                                    },
                                    {
                                        "name": "trouserSizeLength",
                                        "mu": 12.4,
                                        "sigma": None
                                    }
                                ]
                            }

        '''


        # sample_key = 'new_customer:shoeSize:43'

        size_dimensions = [
            'shoeSize',
            'tShirtSize',
            'trousersSizeWidth',
            'trousersSizeLength'
        ]
        #TODO: Should I use the same customer_id as the other functions ?

        size_list = []
        for dim in size_dimensions:
            cust_size = customer_object['profile'][dim]
            model_size = self.redis_store.get_single_new_customer_size(dim, str(cust_size))
            size_list.append(model_size)

        ret_dict = {
            'customerId': customer_object['id'],
            'isFirstTimeCustomer': True,
            'sizes': size_list
        }

        return ret_dict


    def _get_existing_customer_size(self, customer_id):
        '''
        Get the mean & std dev of the size of a specific repeat customer

        Args:
            customer_id: The customer to get sizes for

        Returns:
            dictionary: A Dictionary with the customer_id and that customers sizes. Example:
                            {
                                "customerId": 122345_1224,
                                "isFirstTimeCustomer": False,
                                "shoeSize": {
                                    "mu": 12.4,
                                    "sigma": 3.2
                                },
                                "tShirtSize": {
                                    "mu": 12.4,
                                    "sigma": 3.2
                                },
                                "TrouserSize": {
                                    "mu": 12.4,
                                    "sigma": 3.2
                                }
                            }

        '''

        cust_dct = self.redis_store.get_single_existing_customer_sizes(customer_id)
        cust_dct['isFirstTimeCustomer'] = False
        return cust_dct

