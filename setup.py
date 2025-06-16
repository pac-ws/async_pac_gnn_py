import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'async_pac_gnn_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'lpac = async_pac_gnn_py.lpac:main',
            'lpac_l1 = async_pac_gnn_py.lpac_l1:main',
            'lpac_l2 = async_pac_gnn_py.lpac_l2:main',
            'fake_robot = async_pac_gnn_py.fake_robot:main'
        ],
    },
)
