from setuptools import setup

with open('requirements.txt') as file:
    required = file.read().splitlines()

if __name__ == "__main__":
    setup(
        install_requires=required
    )