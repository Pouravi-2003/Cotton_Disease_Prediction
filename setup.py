import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.1"

REPO_NAME = "CCotton_Disease_Prediction"
AUTHOR = "Pouravi Ghosh"
USER_NAME = "Pouravi-2003"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "pouravighosh@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/Pouravi-2003/Cotton_Disease_Prediction.git",
    project_urls={
        "Bug Tracker": f"https://github.com/{USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)