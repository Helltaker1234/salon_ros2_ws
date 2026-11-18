from setuptools import find_packages
from setuptools import setup

setup(
    name='physics_atv_visual_mapping',
    version='0.0.0',
    packages=find_packages(
        include=('physics_atv_visual_mapping', 'physics_atv_visual_mapping.*')),
)
