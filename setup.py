from setuptools import setup, find_packages
package_data = { '': ['*.so*']}
setup(name='akg',
      version='1.0',
      description='akg python libs',
      package_data=package_data,
      packages=find_packages())
