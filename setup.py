# coding:utf-8
import setuptools

with open("ReadMe.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aroma",
    version="0.0.1",
    author="L_Zealot",
    author_email="zhenningli91@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)