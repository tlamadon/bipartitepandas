import setuptools

with open('README.rst', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='bipartitepandas',
    version='0.0.5',
    author='Thibaut Lamadon',
    author_email='thibaut.lamadon@gmail.com',
    description='Python tools for bipartite labor data',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/tlamadon/bipartitepandas',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'numpy_groupies',
        'pandas',
        'scipy',
        'scikit-learn',
        'networkx',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
