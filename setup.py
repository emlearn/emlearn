from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os.path
import setuptools

__version__ = '0.9.0'

project_dir = os.path.abspath(os.path.dirname(__file__))

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)



# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'eml_trees',
        ['bindings/eml_trees.cpp'],
        include_dirs=[
            'emlearn/',
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'eml_audio',
        ['bindings/eml_audio.cpp'],
        include_dirs=[
            'emlearn/',
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'eml_net',
        ['bindings/eml_net.cpp'],
        include_dirs=[
            'emlearn/',
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'eml_bayes',
        ['bindings/eml_bayes.cpp'],
        include_dirs=[
            'emlearn/',
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'eml_signal',
        ['bindings/eml_signal.cpp'],
        include_dirs=[
            'emlearn/',
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]

def read_requirements():
    requirements_txt = os.path.join(project_dir, 'requirements.txt')
    with open(requirements_txt, encoding='utf-8') as f:
        contents = f.read()

    specifiers = [s for s in contents.split('\n') if s]
    return specifiers

def read_readme():
    readme = os.path.join(project_dir, 'README.md')
    with open(readme, encoding='utf-8') as f:
        long_description = f.read()

    return long_description


setup(
    name='emlearn',
    version=__version__,
    author='Jon Nordby',
    author_email='jononor@gmail.com',
    url='https://github.com/emlearn/emlearn',
    description='Machine learning for microcontrollers and embedded systems',
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=['emlearn'],
    scripts={
        'examples/window-function.py': 'eml-window-function',
        'examples/mel-filterbank.py': 'eml-mel-filterbank',
    },
    ext_modules=ext_modules,
    include_package_data=True,
    package_data = {
        '': ['*.h', '*.ino'],
    },
    install_requires=read_requirements(),
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
