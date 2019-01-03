from abc import ABC, abstractmethod
from sizeapi.utils.json_utils import load_json, fpath
import os
import sizeapi

class BaseDecider(ABC):
    '''
    The base class that any decider must inherit from.

    It requires that the following methods be implemented:
     * validate_request
     * validate_response
     * decide
    '''

    def __init__(self, name):
        '''
            Initialise the decider setting it's name and version
        Args:
            name: The name of the decider used to select items
        '''

        self.name = name
        self.version = self._get_version()

    def _get_version(self):
        '''
        Get the current commit of the git repository

        Returns:
            string: The current commit of the git repository

        '''


        #See: https://stackoverflow.com/questions/32523121/gitpython-get-current-tag-detached-head/32524783
        # repo = git.Repo(search_parent_directories=True)
        # tag = next((tag for tag in repo.tags if tag.commit == repo.head.commit), '0.0.0')
        # tag = str(tag)

        try:
            version = sizeapi.VERSION
        except BaseException:
            pass

        return version

    @abstractmethod
    def validate_request(self, request):
        pass

    @abstractmethod
    def validate_response(self, response):
        pass

    @abstractmethod
    def decide(self, request_data):
        pass

    # @abstractmethod
    # def _repeat_customer_decision(self, request):
    #     pass
    #
    # @abstractmethod
    # def _new_customer_decision(self, request):
    #     pass
