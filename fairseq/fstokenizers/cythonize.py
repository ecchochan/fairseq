#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from Cython.Build import cythonize

from distutils.core import setup, Extension
from distutils.extension import Extension

import sys

sys.argv = [sys.argv[0]] + ['build_ext','--inplace']

#extensions = [Extension("tokenizer",["tokenizer.pyx"])]

setup(ext_modules = cythonize('*.pyx', 
                              compiler_directives={
                                  'boundscheck': False, 
                                  'wraparound':False, 
                                  'initializedcheck': False,
                                  'infer_types': True
                              },annotate=True))


'''
python3 cythonize.py build_ext --inplace
python3.6 cythonize.py build_ext --inplace
python3.5 cythonize.py build_ext --inplace
'''