from setuptools import setup, find_packages

setup(
    name='interview_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "asyncio",
        "nest_asyncio",
        "transformers[torch]",
        "datasets",
        "evaluate",
        "seqeval",
        "bitsandbytes",
        "accelerate",
        "peft",
        "trl",
        "torch",
        "huggingface_hub",
        "tqdm",
        "sentencepiece",
        "protobuf",
        "multion",
        "pillow==9.4.0"
    ],
)