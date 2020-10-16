import os
from setuptools import setup

package_data = {'': ['*.so*']}
include_dirs = ['python/akg',
                'third_party/incubator-tvm/python/tvm',
                'third_party/incubator-tvm/topi/python/topi',
                'tests/fuzz',
                'tests/common']

def find_files(where=['.']):
    """
    Return a package list

    'where' is the root directory list
    """
    dirs = [path.replace(os.path.sep, '.') for path in where]
    for selected_root in where:
        for root, all_dirs, files in os.walk(selected_root, followlinks=True):
            for dir in all_dirs:
                full_path = os.path.join(root, dir)
                package = full_path.replace(os.path.sep, '.')
                if '.' in dir:
                    continue
                dirs.append(package)
    dirs.append('build')
    return dirs

setup(name='akg',
      version='1.0',
      description='akg python libs',
      package_data=package_data,
      packages=find_files(include_dirs))
