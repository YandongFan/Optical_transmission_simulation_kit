from setuptools import setup, find_packages

setup(
    name="optical_simulation_kit",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "PyQt6",
        "numpy",
        "torch",
        "scipy",
        "matplotlib",
        "h5py",
        "pandas"
    ],
    entry_points={
        'console_scripts': [
            'optical-sim=src.gui.main_window:main',
        ],
    },
)
