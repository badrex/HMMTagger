# Parts-of-Speech Tagging based on Hidden Markov Model

This repository contains an elegant implementation of a POS tagger based on Hidden Markov Models. To a better view of the internal structure of the developed modules, check the jupypter [notebook](https://github.com/badrex/HMMTagger/blob/master/HMM_POS_Tagger.ipynb) in this repository.

The HMMTagger class in [HMMTagger.py](https://github.com/badrex/HMMTagger/blob/master/HMMTagger.py) relies on clean and well-document implementation of another class that represents a discrete probability distribution with different smoothing techniques. The probabilistic components of the HMM can smoothed using simple add-one smoothing or more advanced absolute discounting.


### (1) Training a POS Tagger
Assuming that we have a training data in ConLL format (check this [folder](https://github.com/badrex/HMMTagger/tree/master/data) for more info), a POS tagger can be trained as follows:

```
$ python  train_tagger.py \
  -train_dir data/ \
  -train_fileid 'de-train.tt' \
  -smoothing 'abs_disc' \
  -d 0.3 \
  -tagger_file 'POS_tagger'
```

### (2) Using a trained POS Tagger
After training a tagger using the previous step, the tagger object would be saved in disk and can be used to tag a test dataset:

```
$ python  test_tagger.py \
  -test_dir data/ \
  -test_fileid de-test.t \
  -POS_tagger POS_tagger \
  -tagged_output tagged_abs_disc.tt
```

### (3) Evaluating the tagger
The python file [eval.py](https://github.com/badrex/HMMTagger/blob/master/eval.py) can used to quantitatively evaluate the performance of the tagger:

```
$ python  data/de-eval.tt tagged_abs_disc.tt
```

Output:

```
Comparing gold file "data/de-eval.tt" and system file "tagged_abs_disc.tt"

Precision, recall, and F1 score:

  DET 0.9092 0.9761 0.9415
 NOUN 0.8476 0.9835 0.9105
 VERB 0.9605 0.8712 0.9137
  ADP 0.9632 0.9762 0.9697
    . 0.9983 0.9992 0.9987
 CONJ 0.9544 0.8974 0.9250
 PRON 0.9391 0.8309 0.8817
  ADV 0.9234 0.7893 0.8511
  ADJ 0.7993 0.6485 0.7160
  NUM 0.9906 0.7778 0.8714
  PRT 0.8730 0.8730 0.8730
    X 0.2000 0.0909 0.1250

Accuracy: 0.9136

```
