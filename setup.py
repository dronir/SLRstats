from setuptools import setup, find_packages


with open('README') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='SLRstats',
    version='0.1.0',
    description='Compute SLR pulse statistics from ray-tracer result',
    long_description=readme,
    author='Olli Wilkman',
    author_email='olli.wilkman@iki.fi',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
