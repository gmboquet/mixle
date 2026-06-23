"""Distributed LDA topic-model estimation on a self-contained synthetic corpus (Spark).

Mirrors examples_pysp/wikipedia_example.py, but parallelizes the corpus with Spark: the only
difference from the local version is that the encoded documents live in an RDD. A synthetic corpus
is sampled from a known LDADistribution (each topic favoring a contiguous block of the vocabulary),
then recovered. No external corpus required.

Run with a JVM available, e.g.:
    export JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export PYSPARK_PYTHON=/path/to/venv/bin/python
    python examples/examples_spark/wikipedia_example.py
"""
import sys
import time

import numpy as np
from pyspark import SparkContext, SparkConf

import pysp.utils.optsutil as ops
from pysp.stats import *
from pysp.inference import initialize, seq_estimate


def make_corpus(num_topics, num_words, num_docs, words_per_doc, seed=2):
    """Sample integer-encoded documents from a known block-structured LDA model."""
    rng = np.random.RandomState(1)
    block = num_words // num_topics
    topics = []
    for i in range(num_topics):
        p = np.full(num_words, 0.2)
        p[i * block:(i + 1) * block] += 5.0
        topics.append(IntegerCategoricalDistribution(0, p / p.sum()))
    truth = LDADistribution(topics, alpha=np.ones(num_topics) * 0.2,
                            len_dist=CategoricalDistribution({words_per_doc: 1.0}))
    docs = truth.sampler(seed=seed).sample(num_docs)
    return [list(ops.count_by_value(doc).items()) for doc in docs], block


if __name__ == '__main__':
    conf = SparkConf().setAppName('wikipedia_example')
    sc = SparkContext(conf=conf)
    sc._jvm.org.apache.log4j.LogManager.getRootLogger().setLevel(
        sc._jvm.org.apache.log4j.Level.ERROR)

    num_topics = 6
    num_words = 60
    out = sys.stdout

    data, block = make_corpus(num_topics, num_words, num_docs=800, words_per_doc=80)
    out.write('#docs=%d  vocab=%d\n' % (len(data), num_words))

    # The only difference from the local example: the data is an RDD.
    data_cnt = sc.parallelize(data, 4)

    topic_est = IntegerCategoricalEstimator(min_val=0, max_val=num_words - 1, pseudo_count=0.01)
    estimator = LDAEstimator([topic_est] * num_topics, keys=(None, 'topics'), gamma_threshold=1.0e-6)

    model = initialize(data_cnt, estimator, np.random.RandomState(1), 0.1)
    enc_data = seq_encode(data_cnt, model=model)

    dcnt, ll_sum = seq_log_density_sum(enc_data, model)
    old_elob = ll_sum / dcnt
    for kk in range(40):
        t0 = time.time()
        model = seq_estimate(enc_data, estimator, prev_estimate=model)
        dcnt, ll_sum = seq_log_density_sum(enc_data, model)
        elob = ll_sum / dcnt
        if kk % 10 == 0:
            out.write('iteration %3d  E[LB]=%e  delta=%e  dt=%.2fs\n'
                      % (kk + 1, elob, elob - old_elob, time.time() - t0))
        if abs(elob - old_elob) < 1.0e-6:
            break
        old_elob = elob

    for i in np.argsort(-model.alpha):
        log_p = model.topics[i].log_p_vec
        top = np.argsort(-log_p)[:block]
        out.write('topic %d [alpha=%.3f]: %s\n'
                  % (i, model.alpha[i], ', '.join('w%d' % w for w in top)))

    sc.stop()
