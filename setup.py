from setuptools import find_packages, setup

with open("phyber_numint/README.md", "r") as f:
    long_description = f.read()

setup(
    name='Phyber NumInt',
    version='0.1.0',
    description='A simple package to perform numerical integration easily',
    package_dir={'': 'phyber_numint'},
    packages=find_packages(where='phyber_numint'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    author='Lautaro Silbergleit',
    author_email='lautisilbergleit@gmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Aproved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy>=1.20'
    ],
    extra_requires=[
        'twine>=4.0.2'
    ],
    python_requires='>=3.10'
)