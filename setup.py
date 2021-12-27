from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

if __name__ == "__main__":
    setup(
        install_requires=required,
    )