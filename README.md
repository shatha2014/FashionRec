# Instagram Text Analysis - FashionRec

## Overview

This repository contains scripts and programs used to analyze textual data from the Instagram API in the context of automatic intelligent fashion identification, classification and recommendation.

### Modules

- `./clean_data` contains scripts for converting JSON Instagram data to csv,tsv,reduced json files that are suited for processing
- `./data_exploration` contains some descriptive analytics about the data
- `./fasttext_on_spark` contains a scalable implementation of FastText to run on Spark clusters
- `./information_extraction` contains scripts for unsupervised information extraction using semantic/syntactic clustering of text to match it to an ontology/domain data as well as using several external APIs as sources of distant supervision. Also contains scripts for evaluation.
- `./wordvecs` contains scripts training and evaluating word embeddings, as well as normalizing text corpora to be used for training word embeddings.
- `./cnn_classification` contains scripts for training a weakly supervised CNN text classifier and model serving.

### Usage

See individual README.md for each module

## References

If using any of the code in this repsitory in your work or research, please cite: [1](TODO)

### Deep Text Mining of Instagram Data Without Strong Supervision

[1] K.Hammar, S.Jaradat, N.Dokoohaki, M.Matskin [*TODO*](link)

```
@Unpublished{hammar1,
  title={Deep Text Mining of Instagram Data Without Strong Supervision},
  author={Hammar, Kim and Jaradat, Shatha and Dokoohaki, Nima and Matskin, Mihhail},
  booktitle = {Proceedings of the International Conference on Web Intelligence},
  series = {WI '18},
  year = {2018},
  location = {Santiago, Chile},
  note = {unpublished}
}
```

## License

BSD 2-clause, see [LICENSE](./LICENSE)

## Authors/Contributors

Kim Hammar, [kimham@kth.se](mailto:kimham@kth.se)

Shatha Jaradat, [shatha@kth.se](mailto:shatha@kth.se)

Nima Dokoohaki, [nimad@kth.se](mailto:nimad@kth.se)
