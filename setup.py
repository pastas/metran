from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Get the version.
version = {}
with open("metran/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='metran',
    version=version['__version__'],
    description='Python package to perform timeseries analysis of multiple'
                'hydrological time series using a dynamic factor model.',
    long_description=long_description,
    url='https://github.com/pastas/metran',
    author='W.L. Berendrecht',
    author_email='',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Hydrology',
    ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.16.5'
                      'matplotlib>=3.0',
                      'pandas>=1.0',
                      'scipy>=1.1',
                      'numba',
                      'pastas>=0.16.0'],
    packages=find_packages(exclude=[]),
)
