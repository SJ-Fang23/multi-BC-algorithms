from setuptools import setup, find_packages

setup(
    name='baselines-drimer',
    version='0.0.1',
    description="Baselines for drimer", # Replace with a brief description
    long_description=open("README.md").read(), # Optional: use a long description from a file
    long_description_content_type="text/markdown", # Optional: specify the content type of the long description
    packages=find_packages(),
    classifiers=[ # Optional: define some trove classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9', # Specify which Python versions you support
)