from Cython.Distutils import build_ext
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
import os
import psutil

class libsensorio_build(build_ext):
    def run(self):
        build_ext.run(self)

compile_args = ['-O3', '-march=native', '-std=c++14']
link_args = compile_args

core_ext = Extension('libsensorio/core',
                   sources=['libsensorio/src/core.cpp'],
                   extra_compile_args=compile_args,
                   extra_link_args=link_args)

setup(name = 'libsensorio',
      packages = ["libsensorio"],
      version = '0.1.0',
      license = 'GNU GPLv3',
      license_file = "LICENSE.md",
      description = "Sensor IO Controller Library for communication with hardware.",
      long_description = "..",
      author = "Cristian Bicheru",
      author_email = "c.bicheru0@gmail.com",
      maintainer = "Highlander Aerospace",
      maintainer_email ="hiaero@wrdsb.ca",
      url = 'about-blank',
      download_url = 'https://github.com/cristian-bicheru/fast-ta/archive/v0.1.3.tar.gz',
      keywords = [],
      install_requires = [
          'numpy'
      ],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ],
      project_urls = {
        'Source': 'https://github.com/Highlander-Aerospace/thrust-bench-mark-i/tree/master/software/controller/libsensorio',
      },
      cmdclass={'build_ext': libsensorio_build},
      ext_modules=[core_ext])
