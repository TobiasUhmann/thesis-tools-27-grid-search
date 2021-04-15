# Problem

Contrary to expectations, the class embeddings do not fit the embeddings 
of semantically similar words very well. The reason might be that the 
words' discriminative features get lost among the many other words in 
the sentence.

# Experiment

Try min/max pooling for getting the sentence embedding and compare the 
results with the usual mean pooling.

# Setup

## Install SpaCy

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```
