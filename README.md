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
