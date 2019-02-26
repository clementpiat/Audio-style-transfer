from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Training VGG model on audio',
      author='Clement PIAT',
      author_email='clement.piat2@gmail.com',
      license='MIT',
      install_requires=[
        'tensorflow-gpu',
        'keras',
        'librosa',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'wheel',
        'soundfile'
      ],
      zip_safe=False)