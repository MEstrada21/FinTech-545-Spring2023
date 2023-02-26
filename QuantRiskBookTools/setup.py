from setuptools import setup, find_packages

setup(
    name='QuantRiskBookTools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'matplotlib',
        # Add any other required packages here
    ],
    author='Your Name',
    description='Description of your package',
    url='https://github.com/your_username/your_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)