from setuptools import setup, find_packages

setup(name='mit-6882-hw1',
      version='0.0.1',
      install_requires=['pddlgym @ git+https://github.com/tomsilver/pddlgym#egg=pddlgym'],
      packages=find_packages(),
      include_package_data=True,
)