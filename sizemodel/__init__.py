import os
import json
import subprocess

VERSION = '0.0.0-unknown-version'
"""str: Current, auto-parsed module version.
If this file exists in a git repository, the latest --tag is returned as version.
Else if a local buildcontext.json exists, the version is read from it.
Else the version is '0.0.0-unknown-version'.
"""

COMMIT = 'unknown-commit'
"""str: Current, auto-parsed module git commit hash.
If this file exists in a git repository, the latest HEAD is returned as commit.
Else if a local buildcontext.json exists, the commit is read from it.
Else the commit is 'unknown-commit'.
"""

REPONAME = 'unknown-repo'
"""str: Auto-parsed module git repistory.
If this file exists in a git repository, the remote.origin.url is returned as repository.
Else if a local buildcontext.json exists, the repository is read from it.
Else the repository is 'unknown-commit'.
"""


def fpath():
    """Absolute path to this file."""
    return os.path.dirname(os.path.abspath(__file__))

def load_json(fname):
    """Open and read in a JSON file."""
    with open(fname, 'r') as f:
        return json.load(f)


try:
    _buildcontext = load_json(os.path.join(fpath(), 'buildcontext.json'))
    VERSION = _buildcontext['version']
    COMMIT = _buildcontext['commit']
    REPONAME = _buildcontext['repo']
except BaseException:
    pass


try:
    VERSION = subprocess.check_output(
        ['git', 'describe', '--tags'],
        cwd=fpath()).strip().decode()
except BaseException:
    pass

try:
    COMMIT = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=fpath()).strip().decode()
except BaseException:
    pass


try:
    REPONAME = subprocess.check_output(
        ['git', 'config', '--get', 'remote.origin.url'],
        cwd=fpath()).strip().decode()
except BaseException:
    pass


__version__ = VERSION
