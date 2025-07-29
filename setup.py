from distutils.core import setup

setup(
    name="smilelogging",  # How you named your package folder (MyLib)
    packages=["smilelogging"],  # Chose the same as "name"
    version="0.6",  # Start with a small number and increase it with every change you make
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="Python logging package for easy reproducible experimenting in research.",  # Give a short description about your library
    author="Huan Wang",  # Type in your name
    author_email="wanghuan@westlake.edu.cn",  # Type in your E-Mail
    url="https://github.com/MingSun-Tse/smilelogging",  # Provide either the link to your github or to your website
    download_url="https://github.com/MingSun-Tse/smilelogging/archive/refs/tags/v0.6.tar.gz",  # I explain this later on
    keywords=[
        "logging",
        "reproducible experimenting",
    ],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        "numpy",
        "pandas",
        "matplotlib",
        "Pillow",
        "PyYAML",
        "lmdb",
        "configargparse",
        "pytz",
        "colorama",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Science/Research",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3",  # Specify which Python versions that you want to support
    ],
)
