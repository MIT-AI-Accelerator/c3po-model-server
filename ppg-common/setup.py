from setuptools import find_packages, setup

setup(name='ppg-common',
      version='1.9.0',
      description='A library for PPG common code',
      url='--',
      author='MIT Lincoln Laboratory',
      license='MIT',
      packages=['ppg.core',
		'ppg.schemas',
		'ppg.schemas.bertopic',
		'ppg.schemas.gpt4all',
		'ppg.schemas.mattermost',
		'ppg.services'],
      include_package_data=True,
      install_requires = [],
      zip_safe=True)
