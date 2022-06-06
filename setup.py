from setuptools import find_packages, setup

def read_requirements(path):
    with open(path, 'r') as file:
        return file.readlines()

setup(
    name="paysim_analysis",
    python_requires='>3.8',
    version="0.0.1",
    description="project_description",
    url="https://github.com/fabiana001/paysim_analysis",
    long_description="Exploratory Data Analysis",
    author="fabiana lanotte",
    package_dir={'': '.'},  # Optional
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt")
)

