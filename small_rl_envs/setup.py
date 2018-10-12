from setuptools import setup, find_packages

long_description = """
Collection of simple OpenAI gym environments to train and test your reinforcement learning algorithms
"""


setup(
    name='small-rl-envs',
    version='0.1.0',
    description="Small reinforcement learning environments",
    long_description=long_description,
    url='https://github.com/MillionIntegrals/small-rl-envs',
    author='Jerry Tworek',
    author_email='jerry@millionintegrals.com',
    license='MIT',
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'gym',
        'numba'
    ],
    extras_require={
        'dev': ['nose']
    },
    entry_points={
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    scripts=[]
)

