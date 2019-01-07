import jsonschema
from jsonschema import Draft4Validator as Validator
import logging

# from tinder.db.db import fetch_from, get_sql
# from tinder.db.connections import mldb as db_conn
from sizemodel.sizeapi.deciders.decider_base import BaseDecider
from sizemodel.sizeapi.utils.json_utils import load_json_resource
# from tinder.db.db import insert_into_table_batched

# TODO: make application_id into enum ?
SizeModelRequest = load_json_resource(__name__, 'resources/SizeModelRequest.json')
SizeModelResponse = load_json_resource(__name__, 'resources/SizeModelResponse.json')

log = logging.getLogger(__name__)


class ATLSizeModelDecider(BaseDecider):
    '''
    A decider that simply returns random items from current stock that the user has not given feedback on before
    '''

    def __init__(self):
        super().__init__('atl_sizemodel_decider')

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
            else self._get_repeat_customer_size(request_data['attributes']['customerId'])

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

        #TODO: Actually implement

        retlist = []
        for article in article_ids:
            retlist.append({'articleId': article, 'size': {'mu': 12.4, 'sigma': 3.2}})

        return retlist

    def _is_new_customer(self, request_data):
        '''
        Is a customer new or repeat

        Args:
            request_data: The request data sent to the API

        Returns:
            bool: Whether or not the given customer is a new customer
        '''

        #TODO: Implement
        return False

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
                                        "sigma": 3.2
                                    },
                                    {
                                        "name": "tShirtSize",
                                        "mu": 12.4,
                                        "sigma": 3.2
                                    },
                                    {
                                        "name": "trouserSizeWidth",
                                        "mu": 12.4,
                                        "sigma": 3.2
                                    },
                                    {
                                        "name": "trouserSizeLength",
                                        "mu": 12.4,
                                        "sigma": 3.2
                                    }
                                ]
                            }

        '''

        # TODO: Actually implement
        customer_id = customer_object['id']

        dct = {
            "customerId": customer_id,
            "isFirstTimeCustomer": True,
            "sizes": [
                {
                    "name": "shoeSize",
                    "mu": 12.4,
                    "sigma": 3.2
                },
                {
                    "name": "tShirtSize",
                    "mu": 12.4,
                    "sigma": 3.2
                },
                {
                    "name": "trouserSizeWidth",
                    "mu": 12.4,
                    "sigma": 3.2
                },
                {
                    "name": "trouserSizeLength",
                    "mu": 12.4,
                    "sigma": 3.2
                }
            ]
        }


        return dct


    def _get_repeat_customer_size(self, customer_id):
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

        #TODO: Actually implement

        return {'customerId': customer_id, 'size': {'mu': 12.4, 'sigma': 3.2}}

