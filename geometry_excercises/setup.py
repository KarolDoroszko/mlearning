import shutil
import setuptools
from geometry_excercises import version

with open('.\README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = [line for line in f if '--' not in line]

setuptools.setup(
    name='geometry_excercises',
    version=version,
    author='Karol Doroszko',
    author_email='doroszko.karol@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='git_url',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming language :: Pyhton :: 3',
        'Operating System :: Microsoft :: Windows',
    ],
    dependency_links=[
        'http://IP',
    ],
    install_requirements=requirements,
    python_requires='>=3.7',
)