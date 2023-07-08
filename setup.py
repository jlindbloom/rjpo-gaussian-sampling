from setuptools import setup, find_packages

setup(
    name='rjpo',
    version='0.0.1',
    author='Jonathan Lindbloom',
    author_email='jonathan@lindbloom.com',
    license='LICENSE',
    packages=find_packages(),
    description='A Python implementation of a reversible jump perturbation optimization (RJPO) method for sampling high-dimensional Gaussians.',
    long_description=open('README.md').read(),
)