from setuptools import setup, find_packages

setup(
    name="text_dataset",
    version="0.1.29",
    author="Eliton Perin",
    author_email="elitonperin@gmail.com",
    description="Wrapper para centralizar e facilitar o acesso a datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/liafacom/text_dataset",  # Substitua pelo repositório do GitHub
    packages=find_packages(),
    install_requires=[
        "pandas",  # Dependências principais
        "numpy",
        "nltk",
        "scikit-learn",
        "tqdm",
        "kagglehub",
        "datasets",  # Biblioteca para manipulação de datasets
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,  # Inclui arquivos não-Python
    package_data={
        "text_dataset": [
            "artifacts/*",
            "datasets_path/*.csv",
            "datasets_path/*.meta",
            "datasets_path/*.txt",
            "datasets_path/data-web-snippets/*.txt",
            "datasets_path/mpqa/*.txt",
            "datasets_path/sst2/*.tsv",
            "datasets_path/sst5/*.csv",
            "datasets_path/per_sen_t/*.csv",
            "datasets_path/iSarcasm/*.csv",
            "datasets_path/tag_my_news/business/*.txt",
            "datasets_path/tag_my_news/entertainment/*.txt",
            "datasets_path/tag_my_news/health/*.txt",
            "datasets_path/tag_my_news/sci_tech/*.txt",
            "datasets_path/tag_my_news/sport/*.txt",
            "datasets_path/tag_my_news/us/*.txt",
            "datasets_path/tag_my_news/world/*.txt",
        ],  # Inclui os arquivos de datasets_path
    },
)
