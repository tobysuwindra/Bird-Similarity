import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='Bird-Similarity',
    version='0.0.1',
    author='Toby Suwindra',
    author_email='tsuwindra@gmail.com',
    description='This model has been trained on Resnet50 and tested on AUC with 93% accuracy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tobysuwindra/Bird-Similarity",
    packages=setuptools.find_packages(),
    install_requires=['torch', 'torchvision'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)