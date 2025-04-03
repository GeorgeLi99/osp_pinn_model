from setuptools import setup, find_packages

setup(
    name="lumerical",
    version="2.4.2",
    description="Lumerical Python API",
    packages=find_packages(),
    py_modules=["lumapi", "lumjson", "lumslurm", "add_partitions"],
    include_package_data=True,
)
