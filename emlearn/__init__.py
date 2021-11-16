
from . import trees
from . import common
from . import signal
from . import tools

from .convert import convert

def get_version():
    import os.path
    here = os.path.dirname(__file__)
    with open(os.path.join(here, 'VERSION.txt')) as version_file:
        version = version_file.read().strip()
        return version

__version__ = get_version()

includedir = common.get_include_dir()

