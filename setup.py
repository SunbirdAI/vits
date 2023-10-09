from setuptools import setup, find_packages
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='sunbird_vits',
    version='0.0.6alpha-1',
    ext_modules = cythonize("monotic_align/core.pyx"),
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    install_requires=[
        # "Cython==0.29.21",
        # "librosa==0.8.0",
        # "matplotlib==3.3.1",
        # #"numpy",
        # "phonemizer==2.2.1",
        # #"scipy==1.5.2",
        # "tensorboard==2.3.0",
        # #"torch==1.6.0",
        # #"torchvision==0.7.0",
        # "Unidecode==1.1.1"

    ],
    entry_points={
        'console_scripts': [
            # If you want to create any executable scripts.
            # 'script_name = my_package.some_module:main_func',
        ],
    },
    license='CC-sharealike-2.0',
    description='A reworking of the vits package with additional proceesing for sunbirdAI data',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/SunbirdAI/vits',
    author='Ali Hussein',
    author_email='azawahry@sunbird.ai',
)


  
  
