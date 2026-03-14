from setuptools import setup, find_packages

setup(
    name='Disaster Agent',
    version='0.1.0',
    author='Ankit Basu',
    author_email='ankitbasu@tamu.edu',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)