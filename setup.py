"""Install Processing Swarm."""
from setuptools import setup

setup(
    name='processing_swarm',
    entry_points={
        'console_scripts': [
            'processing_swarm = processing_swarm:main',
        ],
    }
)
