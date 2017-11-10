"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import nltk
import string

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  #with tf.gfile.GFile(filename, "r") as f:
  #  if Py3:
  #    return f.read().replace("\n", "<eos>").split()
  #  else:
  #    return f.read().decode("utf-8").replace("\n", "<eos>").split()
  training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=filename,
      target_dtype=str,
      features_dtype=str)
  d = {}
  for entry in zip(training_set.target,training_set.data):
    if entry[0] in d:
      d[entry[0]].append(entry[1])
    else:
      d[entry[0]] = [ entry[1] ]
  EAP_words = []
  for s in d['EAP']:
    EAP_words += [ word.lower() if word != '.' else '<eos>' for word in nltk.word_tokenize(s[1]) ]
    #EAP_words += [ word for word in tok if word not in string.punctuation ]
  MWS_words = []
  for s in d['MWS']:
    MWS_words += [ word.lower() if word != '.' else '<eos>' for word in nltk.word_tokenize(s[1]) ]
    #MWS_words += [ word for word in tok if word not in string.punctuation ]
  HPL_words = []
  for s in d['HPL']:
    HPL_words += [ word.lower() if word != '.' else '<eos>' for word in nltk.word_tokenize(s[1]) ]
    #HPL_words += [ word for word in tok if word not in string.punctuation ]
  tr = { 'EAP':EAP_words, 'MWS':MWS_words, 'HPL':HPL_words }
  return tr
    

id_to_word = {}
def _build_vocab(filename):
  data = _read_words(filename)

  all_vocab = data['EAP'] + data['MWS'] + data['HPL']
  counter = collections.Counter(all_vocab)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  #for key, value in word_to_id.items():
  #  id_to_word[value] = key

  return word_to_id,data


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "train_trunc.csv")#"ptb.train.txt")
  #valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "test_trunc.csv")

  word_to_id,tr = _build_vocab(train_path)
  train_data = {}
  train_data['EAP'] = [word_to_id[word] for word in tr['EAP'] if word in word_to_id]
  train_data['MWS'] = [word_to_id[word] for word in tr['MWS'] if word in word_to_id]
  train_data['HPL'] = [word_to_id[word] for word in tr['HPL'] if word in word_to_id]#_file_to_word_ids(train_path, word_to_id)
  #valid_data = _file_to_word_ids(valid_path, word_to_id)
  #test_data = _file_to_word_ids(test_path, word_to_id)
  test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=test_path,
      target_dtype=str,
      features_dtype=str,
      target_column=1)
  test_dataw = []
  for s in test_set.target:
    test_dataw.append( [ word.lower() if word != '.' else '<eos>' for word in nltk.word_tokenize(s) ] )
    test_dataw[-1].insert(0, '<eos>')
  test_data = []
  for s in test_dataw:
    test_data.append( [ word_to_id[word] for word in s if word in word_to_id ] )
  vocabulary = len(word_to_id)
  return train_data, test_data, vocabulary#valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    rd = []
    a = 'EAP' #'EAP', 'MWS', 'HPL'
    
    batch_len = len(raw_data[a]) // batch_size
    rd += raw_data[a][0 : batch_size * batch_len]
    
    raw_data = tf.convert_to_tensor(rd, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                        [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y, batch_size