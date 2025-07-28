"""
Setup script for ROMS Grid Tools
"""

from setuptools import setup, find_packages
import os

# Read the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="roms-grid-tools",
    version="0.1.0",
    author="ROMS Grid Tools Contributors",
    author_email="",
    description="Python tools for working with ROMS nested grids",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hetland/roms_grid_tools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
        ],
        'plotting': [
            'cartopy>=0.20.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'roms-contact=roms_grid_tools.contact:main',
        ],
    },
    include_package_data=True,
    package_data={
        'roms_grid_tools': ['*.py'],
    },
)
