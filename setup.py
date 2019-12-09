import setuptools

with open('README.md') as infile:
    long_description = infile.read()

with open('requirements.txt') as infile:
    required = [line.strip() for line in infile]

with open('VERSION') as infile:
    version = infile.read().strip()

setuptools.setup(
    name='pysynth',
    version=version,
    description='Dataset synthesis for Python',
    long_description=long_description,
    long_description_content_type='text/markdown; charset=UTF-8',
    author='Jan Šimbera',
    author_email='simbera.jan@gmail.com',
    python_requires='>=3.7.0',
    url='https://github.com/simberaj/pysynth',
    packages=setuptools.find_packages(exclude=('tests', )),
    install_requires=required,
    extras_require={},
    include_package_data=True,
    license='MIT',
    keywords='synthesis ipf data python',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    zip_safe=True
)
