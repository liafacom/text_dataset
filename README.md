# Text Dataset Wrapper

Esta biblioteca Python é um wrapper para diversos datasets de texto, permitindo obter os dados já divididos em conjuntos de treinamento e teste, juntamente com os nomes dos rótulos e do dataset.

## Instalação

```bash
pip install git+https://github.com/liafacom/text_dataset.git
```

## Uso

### Importação de um Dataset

Cada dataset pode ser carregado utilizando as funções específicas:

```python
from text_dataset.reader import get_r8

df_train, df_test, labels_name, dataset_name = get_r8()
```

### Datasets Disponíveis

Os seguintes datasets podem ser carregados:

- `get_ohsumed`
- `get_r8`
- `get_r52`
- `get_20newsgroups`
- `get_snippets`
- `get_mr`
- `get_agnews`
- `get_sst2`

### Obtendo um Dataset Reduzido

Para obter uma porção menor de um dataset, a função `get_tiny_dataset` pode ser utilizada:

```python
from text_dataset.reader import get_tiny_dataset

df_train_tiny, df_test_tiny, target_names, dataset_name = get_tiny_dataset(df_train, df_test, max_sample_class=10, random_state=42, keep_test=False)
```

#### Parâmetros da Função `get_tiny_dataset`

- `df_train`: DataFrame contendo os dados de treinamento.
- `df_test`: DataFrame contendo os dados de teste.
- `max_sample_class`: Número máximo de amostras por classe no dataset reduzido (padrão: 10).
- `random_state`: Define a semente para garantir reprodutibilidade (padrão: 42).
- `keep_test`: Se `True`, mantém os dados de teste sem alterações; se `False`, remove os dados de teste no dataset reduzido.

## Licença

.

