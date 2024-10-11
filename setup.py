from setuptools import find_packages, setup
from typing import List

HYPHEN_DOT= "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return a list of requirements from the given file.

    It reads the requirements.txt file, removes any empty lines or comments,
    and returns a clean list of package names.
    """
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.strip()
            # Ignore blank lines and comments
            if line and not line.startswith('#'):
                requirements.append(line)
            if HYPHEN_DOT in requirements:
                requirements.remove(HYPHEN_DOT)
    return requirements


setup(
    name="mlops",
    version="0.0.1",  # You can set your version here
    author="Lande",
    author_email="olanrewajutokun@gmail.com",
    packages=find_packages(),  # Corrected from 'package_data' to 'packages'
    install_requires=get_requirements('requirements.txt')
)
