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

- `get_ohsumed` - medical
- `get_r8` - news
- `get_r52` - news
- `get_20newsgroups` - news 
- `get_agnews` - news 
- `get_cstr` - computing
- `get_dblp` - scientific
- `get_snippets` - questions
- `get_mpqa` - questions
- `get_TREC6` - questions
- `get_overruling` - law
- `get_mr` - sentiment analysis
- `get_sst2` - sentiment analysis
- `get_sst5` - sentiment analysis
- `get_persent` - sentiment analysis
- `get_poem_sentiment` - sentiment analysis
- `get_twitter_airline_sentiment` - sentiment analysis
- `get_isarcasm` - sarcasm

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

