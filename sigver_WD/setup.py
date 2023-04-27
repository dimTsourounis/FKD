from distutils.core import setup
from setuptools import find_packages
import os
import codecs

setup_path = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(setup_path, 'README.md'), encoding='utf-8-sig') as f:
    README = f.read()

setup(name='sigver',
      version='1.0',
      url='https://github.com/dimTsourounis/FKD_sigver',
      maintainer='Dimitrios Tsourounis',
      maintainer_email='dtsourounis@upatras.gr',
      description='Signature verification package for Feature-based Knowledge Distillation'
                  'training writer-dependent classifiers.',
      long_description=README,
      author='Dimitrios Tsourounisn',
      author_email='dtsourounis@upatras.gr',
      license='BSD 3-clause "New" or "Revised License"',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      install_requires=[
          'numpy>=1.10.4',
          'torch>=0.4.1',
          'torchvision>=0.2.1',
          'scikit-learn>=0.19.0',
          'matplotlib>=2.0',
          'tqdm',
          'scikit-image',
          'visdom_logger',
          'onnx',
          'onnxruntime',
          'onnxruntime-gpu'
      ],
      python_requires='>=3',
      
      packages=find_packages())
