from setuptools import setup

test_requires = [
    'pytest>=5.0.1'
]

dev_requires = [
    'flake8==3.7.8',
    'flake8-quotes>=2.1.0'
] + test_requires

setup(
    name='openpose-cycling',
    version='0.1.0',
    packages=['openpose_cycling'],
    url='https://github.com/maxboiten/openpose-cycling',
    license='MIT',
    author='Floris Hoogenboom & Max Boiten',
    author_email='floris@digitaldreamworks.nl',
    description='An application of pose estimation to cycling images.',
    install_requires=[
        'keras>=2.2.4',
        'opencv-python>=3.0',
        'numpy>=1.15.1'
    ],
    test_requires=test_requires,
    extras_require={
        'test': test_requires,
        'dev': dev_requires
    }
)