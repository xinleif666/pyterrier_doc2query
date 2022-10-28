# PyTerrier_doc2query

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [docTTTTTquery](https://github.com/castorini/docTTTTTquery) approach for document expansion by query prediction [Nogueira20].

## Installation

This repostory can be installed using Pip.

    pip install --upgrade git+https://github.com/terrierteam/pyterrier_doc2query.git

## What does it do?

A Doc2Query object has a transform() function, which takes the text of each document, and suggests questions
for that text. 

```python

sample_doc = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated"

import pyterrier_doc2query
import pandas as pd
doc2query = pyterrier_doc2query.Doc2Query()
doc2query.transform(pd.DataFrame([{"docno" : "d1", "text" : sample_doc]]))

```

The resulting dataframe returned by transform() will have an additional `"querygen"` column, which
contains the generated queries, such as:

| docno | querygen  |
|-------|-----------|
| "d1"  | 'what was the importance of the manhattan project to the united states atom project? what influenced the success of the united states why was the manhattan project a success? why was it important' |

As a PyTerrier transformer, there are lots of ways to introduce Doc2query into a PyTerrier retrieval
process.

By default, the plugin loads [`macavaney/doc2query-t5-base-msmarco`](https://huggingface.co/macavaney/doc2query-t5-base-msmarco), which is a a version of [the checkpoint released by the original authors](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip), converted to pytorch format.
You can load another T5 model by passing another huggingface model name (or path to model on the file system) by passing it as the first argument:

```python
doc2query = pyterrier_doc2query.Doc2Query('some/other/model')
```

## Using Doc2Query for Indexing


Then, indexing is as easy as instantiating the Doc2Query object and a PyTerrier indexer:

```python

dataset = pt.get_dataset("irds:vaswani")
import pyterrier_doc2query
doc2query = pyterrier_doc2query.Doc2Query(append=True) # append generated queries to the orignal document text
indexer = doc2query >> pt.IterDictIndexer(index_loc)
indexer.index(dataset.get_corpus_iter())
```

## Using Doc2Query for Retrieval

Doc2query can also be used at retrieval time (i.e. on retrieved documents) rather than 
at indexing time.

```python

import pyterrier_doc2query
doc2query = pyterrier_doc2query.Doc2Query()

dataset = pt.get_dataset("irds:vaswani")
bm25 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
bm25 >> pt.get_text(dataset) >> doc2query >> pt.text.scorer(body_attr="querygen", wmodel="BM25")

```

## Examples

Check out out the notebooks, even on Colab:

 - Vaswani [[Github](https://github.com/terrierteam/pyterrier_doc2query/blob/master/pyterrier_doc2query_vaswani.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_doc2query/blob/master/pyterrier_doc2query_vaswani.ipynb)]

## Implementation Details

We use a PyTerrier transformer to rewrite documents by doc2query.

## References

  - [Nogueira20]: Rodrigo Nogueira and Jimmy Lin. From doc2query to docTTTTTquery. https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf
  - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation inInformation Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271

## Credits

- Craig Macdonald, University of Glasgow
- Sean MacAvaney, University of Glasgow
