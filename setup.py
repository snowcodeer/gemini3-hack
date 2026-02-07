from setuptools import setup, find_packages

setup(
    name="vidreward",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "mediapipe",
        "torch",
        "matplotlib",
        "mujoco",
        "gymnasium-robotics",
    ],
)
