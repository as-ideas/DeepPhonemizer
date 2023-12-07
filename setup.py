from setuptools import setup, find_packages

long_description = """DeepPhonemizer is a multilingual grapheme-to-phoneme modeling library that leverages recent deep learning 
technology and is optimized for usage in production systems such as TTS. In particular, the library should
be accurate, fast, easy to use. Moreover, you can train a custom model on your own dataset in a few lines of code.

DeepPhonemizer is compatible with Python 3.6+ and is distributed under the MIT license.

Read the documentation at: https://as-ideas.github.io/DeepPhonemizer/
"""

setup(
    name='deep-phonemizer',
    version='0.0.19',
    author='Christian Schäfer',
    author_email='c.schaefer.home@gmail.com',
    description='Grapheme to phoneme conversion with deep learning.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    license='MIT',
    install_requires=['torch>=1.2.0', 'tqdm>=4.38.0', 'PyYAML>=5.1', 'tensorboard',
                      'certifi>=2022.12.7', 'wheel>=0.38.0', 'setuptools>=65.5.1', 'validators>=0.22.0'],
    extras_require={
        'tests': ['pytest-cov'],
        'docs': ['mkdocs', 'mkdocs-material'],
        'dev': ['bumpversion']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    package_data={'': ['*.yaml']}
)
