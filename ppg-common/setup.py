from setuptools import find_packages, setup

setup(name='ppg-common',
      version='0.2.0',
      description='A library for PPG common code',
      url='--',
      author='MIT Lincoln Laboratory',
      license='MIT',
      packages=['ppg.core',
		'ppg.schemas.bertopic',
		'ppg.schemas.mattermost',
		'ppg.services'],
      include_package_data=True,
      install_requires = [],
      zip_safe=True)
