import hashlib
import json
import logging
import os
import pickle

log = logging.getLogger(__name__)


class PickleCache:
    def __init__(self, *keys):
        cache_key_json = json.dumps(keys, sort_keys=True) \
                             .encode('utf-8')
        self.cache_key_hash = hashlib.md5(cache_key_json).hexdigest()
        cache_dir = os.environ['PS_STORE_EXT']
        self.fetch_file = os.path.join(
            cache_dir, '{hash}.pkl'.format(hash=self.cache_key_hash))

    def check(self):
        return os.path.exists(self.fetch_file)

    def invalidate(self):
        if os.path.exists(self.fetch_file):
            os.remove(self.fetch_file)

    def get(self):
        with open(self.fetch_file, 'rb') as f:
            obj = pickle.load(f)
            log.info('Successfully loaded cache {}'.format(self.fetch_file))
            return obj

    def put(self, obj):
        with open(self.fetch_file, 'wb') as f:
            pickle.dump(obj, f)
            log.info('Dumped cache to {}'.format(self.fetch_file))
