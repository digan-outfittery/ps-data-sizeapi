from pkg_resources import resource_filename
import simplejson
import json
import os

def load_json_resource(file_loc, relative_filename):
    fn = resource_filename(file_loc, relative_filename)
    with open(fn, 'r') as f:
        return simplejson.load(f)


def load_json(fname):
    """Open and read in a JSON file."""
    with open(fname, 'r') as f:
        return json.load(f)


def fpath():
    """Absolute path to this file."""
    return os.path.dirname(os.path.abspath(__file__))