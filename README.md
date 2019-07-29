# TRL-PBLM
The code base for the article This is a code repository used to generate the results appearing in [Task Refinement Learning for Improved Accuracy and Stability of Unsupervised Domain Adaptation](https://www.aclweb.org/anthology/P19-1591), ACL 2019.

If you use this implementation in your article, please cite :)
```bib
@inproceedings{ziser-reichart-2019-task,
    title = "Task Refinement Learning for Improved Accuracy and Stability of Unsupervised Domain Adaptation",
    author = "Ziser, Yftah  and
      Reichart, Roi",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1591",
    pages = "5895--5906",
    abstract = "Pivot Based Language Modeling (PBLM) (Ziser and Reichart, 2018a), combining LSTMs with pivot-based methods, has yielded significant progress in unsupervised domain adaptation. However, this approach is still challenged by the large pivot detection problem that should be solved, and by the inherent instability of LSTMs. In this paper we propose a Task Refinement Learning (TRL) approach, in order to solve these problems. Our algorithms iteratively train the PBLM model, gradually increasing the information exposed about each pivot. TRL-PBLM achieves stateof- the-art accuracy in six domain adaptation setups for sentiment classification. Moreover, it is much more stable than plain PBLM across model configurations, making the model much better fitted for practical use.",
}
```

## INSTALLATION

PBLM requires the following packages:

Python >= 2.7.

numpy

scipy

Theano/tensorflow

keras

scikit-learn

