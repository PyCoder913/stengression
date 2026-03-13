from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stengression",
    version="0.0.1",
    author="Rajdeep Pathak, Tanujit Chakraborty",
    author_email="pathakrajdeep91@gmail.com",
    description="Deep spatiotemporal engression networks for probabilistic forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm"
    ],
)
