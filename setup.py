from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ldv_intensity_reconstruction",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for LDV intensity field reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/LDV-Bessel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest", "flake8", "black"],
    },
)