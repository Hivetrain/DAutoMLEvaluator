from setuptools import setup, find_packages

setup(
    name="evaluator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'tqdm',
        'deap',
        'numpy',
        #'dml'  # Make sure this is the correct name of your DML package
    ]
)
