"""Setup script for failure-to-mix package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="failure-to-mix",
    version="1.0.0",
    author="Biostate AI",
    author_email="research@biostate.ai",
    description="LLM Probability Calibration Experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BiostateAIresearch/failure-to-mix",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "aiohttp>=3.8.0",
        "tqdm>=4.62.0",
        "nest_asyncio>=1.5.0",
    ],
    extras_require={
        "drive": ["google-api-python-client>=2.0.0"],
        "dev": ["pytest>=6.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "run-experiments=scripts.run_experiments:main",
        ],
    },
)
