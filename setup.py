from setuptools import setup, find_packages

setup(
    name="jittorsimpletrans",
    version="0.1.0",
    author="Jittor Team",
    author_email="jittor@example.com",
    description="Implementation of Vision Transformer models using Jittor",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/Jittor-SimplTrans",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "jittor>=1.3.0",
        "loguru",
        "tqdm",
    ],
)
