import pathlib
from setuptools import find_packages
from setuptools import setup

package_name = "cornerhex"
here = pathlib.Path(__file__).absolute().parent


def read_version():
    with (here / package_name / '__init__.py').open() as fid:
        for line in fid:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


setup(
    name=package_name,

    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="science physics mathematics visualization",

    url="https://github.com/stammler/cornerhex/",
    project_urls={"Source Code": "https://github.com/stammler/cornerhex/",
                  "Documentation": "https://cornerhex.rtfd.io/"
                  },

    author="Sebastian Stammler",
    author_email="sebastian.stammler@gmail.com",
    maintainer="Sebastian Stammler",

    version=read_version(),
    license="BSD",

    classifiers=["Development Status :: 4 - Beta",
                 "Environment :: Console",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Education",
                 "Intended Audience :: Information Technology",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: BSD License",
                 "Natural Language :: English",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python",
                 "Programming Language :: Python :: 3 :: Only",
                 "Topic :: Education",
                 "Topic :: Scientific/Engineering",
                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "Topic :: Scientific/Engineering :: Information Analysis",
                 "Topic :: Scientific/Engineering :: Mathematics",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Visualization"
                 ],

    packages=find_packages(),
    install_requires=["matplotlib", "numpy", "scipy"],
    include_package_data=True,
    zip_safe=False,
)
