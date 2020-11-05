import os
import subprocess

from setuptools import setup, find_packages
from distutils.command.install import install
from setuptools.command.develop import develop

from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))


def get_description():
    from codecs import open
    # Get the long description from the README file
    with open(os.path.join(here, 'readme.md'), encoding='utf-8') as f:
        return f.read()


def get_version():
    import sys
    sys.path.insert(0, os.path.join(here, "src", "sltp"))
    import version
    v = version.get_version()
    sys.path = sys.path[1:]
    return v


def main():
    setup(
        name='sltp',
        python_requires='>=3.6.9',
        version=get_version(),
        description='The SLTP Generalized Planning Framework: Sample, Learn, Transform & Plan',
        long_description=get_description(),
        url='https://github.com/rleap-project/d2l',
        author='Blai Bonet and Guillem Francès',
        author_email='-',

        keywords='planning logic STRIPS generalized planning',
        classifiers=[
            'Development Status :: 3 - Alpha',

            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',

            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],

        packages=find_packages('src'),  # include all packages under src
        package_dir={'': 'src'},  # tell distutils packages are under src

        install_requires=[
            'setuptools',
            'psutil',
            'bitarray',
            'natsort>=7.0.1',
            'numpy',
            "tarski@git+git://github.com/aig-upf/tarski.git@2d57d46#egg=tarski",
        ],

        # ext_modules=[
        #     CMakeExtension('featuregen', os.path.join(here, "src", "features")),
        # ],
        # cmdclass={'build_ext': BuildExt},
        cmdclass={
            'develop': SltpDevelop,
            'install': SltpInstall,
        },

        extras_require={
            'dev': ['pytest', 'tox'],
            'test': ['pytest', 'tox'],
        },
    )


def _post_install(directory):
    config = 'Release'
    cmake_args = [
        # f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={ext.path}',
        f'-DCMAKE_BUILD_TYPE={config}'
    ]

    build_args = ['--config', config, '--', '-j4']
    cpp_src_path = os.path.join(here, 'src/generators')
    subprocess.call(['cmake', '.'] + cmake_args, cwd=cpp_src_path)
    subprocess.call(['cmake', '--build', '.'] + build_args, cwd=cpp_src_path)


class SltpInstall(install):
    def run(self):
        """Post-installation for installation mode."""
        install.run(self)
        self.execute(_post_install, (self.install_lib,), msg=f"Building SLTP binaries (devel) on {self.install_lib}")


class SltpDevelop(develop):
    def run(self):
        """Post-installation for development mode."""
        develop.run(self)
        self.execute(_post_install, (self.install_lib,), msg=f"Building SLTP binaries on {self.install_lib}")


if __name__ == '__main__':
    main()
