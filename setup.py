#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import os

from setuptools import find_packages, setup

exec(open('randomsubgroups/_version.py').read())
ver_file = os.path.join('randomsubgroups', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'random-subgroups'
DESCRIPTION = "Machine learning with Subgroup Discovery",
LONG_DESCRIPTION = 'A package based on scikit learn that uses subgroup discovery for machine learning.'
# with codecs.open('README.rst', encoding='utf-8-sig') as f:
#     LONG_DESCRIPTION = f.read()
MAINTAINER = 'C. Rebelo Sa'
MAINTAINER_EMAIL = 'c.f.pinho.rebelo.de.sa@liacs.leidenuniv.nl'
URL = 'https://github.com/rebelosa/random-subgroups'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/rebelosa/random-subgroups'
VERSION = __version__
INSTALL_REQUIRES = ['pandas',
                    'numpy',
                    'scikit-learn',
                    'pysubgroup']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
