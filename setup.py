import os
import sys
import json
import shutil
import subprocess
import importlib.util
import pkg_resources
from setuptools import Command, find_packages, setup

# configure these
PACKAGE_NAME = 'ps-data-sizeapi'
MODULE_NAME = 'sizemodel'


# general functions

def load_json(path):
    with open(path) as f:
        return json.load(f)


def fpath():
    """Absolute path to this file."""
    return os.path.dirname(os.path.abspath(__file__))


def mspath():
    """Absolute path to the module's source folder."""
    return os.path.join(fpath(), MODULE_NAME)


def load_build_json():
    """Load the build.json from the root directory of this repo.

    build.json defines various variables, e.g. the specific package name, e.g. acdc, app,
    base_image, models and more.

    build.json should previously been copied there by

       python setup.py prepare_build --build-json=<REL_PATH_TO_BUILD_JSON>
    """
    build_json = os.path.join(fpath(), 'build.json')
    return load_json(build_json)


# automated version discovery

def load_module_from_path(path, name="unknown_module"):
    """Load a python module dynamically based on a python file.
    Paramters
    ---------
    path : str
        Absolute path to python file.
    name : str, optional
        The name to give the module.
    Returns
    -------
    mod : module
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_buildcontext_filename():
    return os.path.join(mspath(), 'buildcontext.json')


def write_buildcontext():
    """Write the build context into a file."""
    autoversion = load_module_from_path(os.path.join(mspath(), '__init__.py'), 'autoversion')
    context = {
        'version':  autoversion.VERSION,
        'repo':     autoversion.REPONAME,
        'commit':   autoversion.COMMIT
    }
    with open(get_buildcontext_filename(), 'w') as f:
        json.dump(context, f, indent=4)


def read_version():
    """Automatically try to determine the current moduel version."""
    return load_module_from_path(os.path.join(mspath(), '__init__.py'), 'autoversion').VERSION


# package verification

def listdir(d, exclude=['__pycache__']):
    lst = os.listdir(d)
    lst = [os.path.join(d, e) for e in lst if e not in exclude]
    files = [e for e in lst if os.path.isfile(e)]
    dirs = [e for e in lst if os.path.isdir(e)]
    return dirs, files


def is_init_missing(root):
    """Ensures recursively that each directory containing a python file also contains an __init__.py file."""
    dirs, files = listdir(root)
    has_init = '__init__.py' in {os.path.basename(file) for file in files}
    has_pyfiles = '.py' in {os.path.splitext(file)[1] for file in files}
    init_missing_in_subpackage = False
    for d in dirs:
        init_missing_in_subpackage = (is_init_missing(d) or
                                      init_missing_in_subpackage)
    init_needed = has_pyfiles or init_missing_in_subpackage
    if init_needed and not has_init:
        sys.stderr.write('__init__.py missing in {}\n'.format(root))
        return True
    if init_missing_in_subpackage:
        return True
    return False


# package build preparation

def copy(src, dst):
    print('Copy "{}" to "{}"'.format(src, dst))
    assert os.path.exists(src), "'{}' not found.".format(src)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.isdir(src):
        shutil.copytree(src, dst)
    elif os.path.isfile(src):
        shutil.copyfile(src, dst)


def get_store_directory():
    storage_folder = os.path.join(mspath(), '_store')
    return storage_folder


def recreate_store_directory():
    storage_folder = get_store_directory()
    # print('Clearing directory {}'.format(storage_folder))
    # shutil.rmtree(storage_folder, ignore_errors=True)
    os.makedirs(storage_folder, exist_ok=True)
    return storage_folder


# def all_models_are_in_store():
#     build_json = load_build_json()
#     storage_folder = get_store_directory()
#
#     # iterate over all models in build.json and check whether it is in the store
#     for model_name in build_json['models']:
#         dst = os.path.join(storage_folder, model_name)
#         if not os.path.exists(dst):
#             sys.stderr.write('File "{}" missing.\n'.format(dst))
#             return False
#     return True


def merge_scripts(scripts):
    """merge given scripts with scripts in build.json"""
    try:
        build_json = load_build_json()
        scripts_in_build_json = build_json.get('scripts', [])

        return list(set(scripts).union(scripts_in_build_json))
    except:
        return scripts


# commands

class VerifyPackage(Command):
    """Verfiy the package to be installable."""
    description = 'Verify package'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if is_init_missing(mspath()):
            sys.exit(1)


class PrepareBuild(Command):
    """Prepare a build (build inside a docker container).

    Copy a project's build.json to
        - the root directory, so it can be used by this script as the reference
        - to pandora/ so that it can be used by pandora's scripts
    Check if all models are present in the store.

    """
    description = 'Prepare build'
    user_options = [
        ('build-json=', 'b', 'path to build.json (defaults to the one in the package\'s source dir)'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.build_json = None

    def finalize_options(self):
        """Post-process options."""
        assert self.build_json is not None, '--build-json not specified'
        assert os.path.exists(self.build_json), "supplied file {} does not exist".format({self.build_json})

    def run(self):
        """Run command code."""
        # copy build.json to package root directory
        copy(self.build_json, os.path.join(fpath(), 'build.json'))



class CopyModels(Command):
    description = ('Recreate the model directory and '
                   'copy models specified in build config')
    user_options = [
        ('build-json=', 'b', 'relative path to build.json'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.build_json = None

    def finalize_options(self):
        """Post-process options."""
        assert self.build_json is not None, '--build-json not specified'
        assert os.path.exists(self.build_json), "supplied file {} does not exist".format({self.build_json})

    def run(self):
        storage_folder = recreate_store_directory()
        models_folder = os.environ['MODELS_FOLDER']

        # iterate over all models in build.json and copy to models folder
        for model_name in self.build_json['models']:
            src = os.path.join(models_folder, model_name)
            dst = os.path.join(storage_folder, model_name)
            copy(src, dst)


# installing with dependency links
def add_github_token(uri):
    if os.environ.get('GITHUB_ACCESS_TOKEN'):
        token = os.environ.get('GITHUB_ACCESS_TOKEN') + '@'
    else:
        token = ''
    uri = uri.format(token=token)
    return uri


write_buildcontext()  # always write the build context when this file is called


_setup = setup(
    name=PACKAGE_NAME,
    version=read_version(),
    packages=find_packages(),
    # miminal dependencies for production app
    install_requires=[
        'Flask==0.12',
        'Flask-Cors==3.0.2',
        'Flask-Healthcheck==0.1.2',
        'jsonschema==2.6.0',
        'simplejson==3.10.0',
        'pandas==0.23.4',
        # 'psycopg2==2.7.5',
        # 'sqlalchemy==1.2.12',
        'prometheus_client==0.4.1',
        # 'gitpython==2.1.11',
        'python-logstash==0.4.6',
        # 'newrelic==4.2.0.100',
        # 'gunicorn==19.6.0'
    ],
    dependency_links=[
        # Should ALWAYS resolve a tagged release version!
        # format: git+https://{token}github.com/{acc-name}/{repo-name}.git@{tag}#egg={package-name}-{tag}
        #         with version == tag
        # install_requires: [{package-name}-{tag}]

        # Alternatives
        # format: git+https://{token}github.com/{acc-name}/{repo-name}.git@{short_commit_hash|tag|branch}#egg={package-name}-{version}+git.{short_commit_hash}
        # install_requires: [{package-name}=={version}+git.{short_commit_hash}]

        # Everything that is qritten behind #egg= is an arbitrary name that should be reflected in
        # install_requires The actual package version (as declared in setup.py) should be the same
        # as the #egg= points to to avoid problems note that the versioning is arbitrary, as pip
        # simply pulls what it finds at {repo-name}.git@{commit} savely ignore any "Could not find a
        # tag or branch '...', assuming commit." during installation
        # add_github_token('git+https://{token}github.com/paulsecret/ps-data-joblib.git@1.7.8#egg=ps-data-joblib-1.7.8'),
        # add_github_token('git+https://{token}github.com/paulsecret/ps-data-store.git@1.3.2#egg=ps-data-store-1.3.2')
    ],
    cmdclass={
        'verify': VerifyPackage,
        'prepare_build': PrepareBuild
    },
    extras_require={
        'dev': ['flask==0.12.3',
                'flask-cors==3.0.3'
                ]
    },
    # run
    #   python3 setup.py prepare_build
    # followed by
    #   python3 setup.py bdist
    # then you can check the `build` folder to see if your package data is included in
    # the binary distribution (which is used to build the docker image)
    package_data={
        '': [
            'build.json',
            'buildcontext.json',
            'resources/*',
            'scripts/*',
            'tools/*',
            '{}/*'.format('_store'),
            '{}/**/*'.format('_store')
        ]
    },

    scripts=merge_scripts([
        # 'scripts/start_service.sh',
        # 'scripts/gunicorn_conf.py'
    ]),
)
# add scripts from build.json
