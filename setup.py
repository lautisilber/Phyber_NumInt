from setuptools import find_packages, setup

with open("phyber_numint/README.md", "r") as f:
    long_description = f.read()

setup(
    name='phyber_numint',
    version='0.1.1',
    description='A simple package to perform numerical integration easily',
    package_dir={'': 'phyber_numint'},
    packages=find_packages(where='phyber_numint'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lautisilber/Phyber_NumInt',
    author='Lautaro Silbergleit',
    author_email='lautisilbergleit@gmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    install_requires=[
        'numpy>=1.20'
    ],
    extra_requires=[
        'twine>=4.0.2'
    ],
    python_requires='>=3.7'
)