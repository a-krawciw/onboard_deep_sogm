from setuptools import setup

package_name = 'publish_goal_cost'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alec',
    maintainer_email='freakyfactoid@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nbv_goal_node = publish_goal_cost.nbv_goal_node:main',
            'nbv_training = publish_goal_cost.nbv_training_node:main'
        ],
    },
)
