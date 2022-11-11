from setuptools import setup, find_packages

setup(
    name='ecco_pipeline',
    version='0.1',
    package_dir={'': '.'},
    packages=find_packages(include=['ecco_pipeline', 'ecco_pipeline.*']),
    install_requires=[
        'pyresample',
        'netcdf4',
        'xarray'
    ]
)