from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fr:
    reqs = fr.read().strip().split("\n")

setup(
    name = "spwk_agtech",
    version = "0.1.0",
    author= "hayoung choi",
    author_email = "hychoi@spacewalk.tech",
    description = "Tests for spacewalk Agtech engineer applicants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages = find_packages(),
    include_package_data=True,
    url="https://github.com/TeamSPWK/spwk-agtech-task",
    python_requires=">=3.6",
    install_requires = reqs
)