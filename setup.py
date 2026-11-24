from setuptools import setup, find_packages

setup(
    name="hmp_evaluation",
    version="1.0.0",
    description="A comprehensive framework for evaluating accuracy and realism in Human Motion Prediction (HMP)",
    author="Michael Vanuzzo",
    author_email="michaelvanuzzo@gmail.com",
    url="https://github.com/MichaelVanuzzo/HMP-Evaluation-Framework",
    license="AGPL-3.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "hmp_utils @ git+https://github.com/MichaelVanuzzo/hmp_utils.git",
    ],
    python_requires=">=3.8",
)
