from setuptools import setup, find_packages


with open('README.md', 'r') as fs:
    long_description = fs.read()

setup(
    name='edunet',
    version='0.0.0',
    author='Matas Gumbinas',
    author_email='matas.gumbinas@gmail.com',
    description='Edunet. Numpy based educational neural networks modeling framework - from scratch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.12.0'
    ],
)
