# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides function to build an event sequence RNN model's graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import six
import tensorflow as tf
import magenta

from tensorflow.python.util import nest as tf_nest

def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units)
    if attn_length and not cells:
      # Add attention wrapper to first layer.
      cell = tf.contrib.rnn.AttentionCellWrapper(
          cell, attn_length, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells)

  return cell


def get_build_graph_fn(mode, config, sequence_example_file_paths=None):
  """Returns a function that builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
        to use.
    sequence_example_file_paths: A list of paths to TFRecord files containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation.

  Returns:
    A function that builds the TF ops when called.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError("The mode parameter must be 'train', 'eval', "
                     "or 'generate'. The mode parameter was: %s" % mode)

  hparams = config.hparams
  encoder_decoder = config.encoder_decoder

  tf.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size
  num_classes = encoder_decoder.num_classes
  no_event_label = encoder_decoder.default_event_label

  def build():
    """Builds the Tensorflow graph."""
    inputs, labels, lengths = None, None, None

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams.batch_size, input_size,
          shuffle=mode == 'train')
      '''
      with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        try:
            #while not coord.should_stop():
            for i in range(2):
                e1,e2,e3=sess.run([inputs,labels,lengths])
                #print(num)
                print(e1)
                print(e2)
                #print('hello\n')
                #print(len(e2))
                #print('hello1\n')
                print(e3[0])
                #print(len(e3))
                #print('hello3\n')
        except tf.errors.OutOfRangeError:
            print('Done nothing')
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)'''
    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                           input_size])

    cell = make_rnn_cell(
        hparams.rnn_layer_sizes,
        dropout_keep_prob=(
            1.0 if mode == 'generate' else hparams.dropout_keep_prob),
        attn_length=(
            hparams.attn_length if hasattr(hparams, 'attn_length') else 0))

    initial_state = cell.zero_state(hparams.batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell, inputs, sequence_length=lengths, initial_state=initial_state,
        swap_memory=True)

    outputs_flat = magenta.common.flatten_maybe_padded_sequences(
        outputs, lengths)
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    if mode == 'train' or mode == 'eval':
      labels_flat = magenta.common.flatten_maybe_padded_sequences(
          labels, lengths)

      softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels_flat, logits=logits_flat)
      predictions_flat = tf.argmax(logits_flat, axis=1)

      correct_predictions = tf.to_float(
          tf.equal(labels_flat, predictions_flat))
      event_positions = tf.to_float(tf.not_equal(labels_flat, no_event_label))
      no_event_positions = tf.to_float(tf.equal(labels_flat, no_event_label))

      # Compute the total number of time steps across all sequences in the
      # batch. For some models this will be different from the number of RNN
      # steps.
      def batch_labels_to_num_steps(batch_labels, lengths):
        num_steps = 0
        for labels, length in zip(batch_labels, lengths):
          num_steps += encoder_decoder.labels_to_num_steps(labels[:length])
        return np.float32(num_steps)
      num_steps = tf.py_func(
          batch_labels_to_num_steps, [labels, lengths], tf.float32)

      #compute a extra loos we defined according to the music rules
      def extra_loss():
          max_steps = 19200
          bar_distance = 16 * 3
          first_bar = predictions_flat[:max_steps]
          second_bar = predictions_flat[1:max_steps+1]

          result = tf.subtract(second_bar, first_bar)
          bar_with_first = result[:max_steps - bar_distance]
          bar_with_last = result[bar_distance:]

          def neg_to_non_neg(bar):
              '''
              this function is to transform the whole number in a tensor of a bar to non-negative number
              :param bar: the bar which is to be treated
              :return: a tensor which refers to the input bar but all the negative number in the tensor will be 0, and
              the positive number will be 1, 0 will not change
              '''
              bar_abs = tf.abs(bar)
              bar_final = tf.add(bar_abs, bar)
              bar_final = tf.div(bar_final, 2)

              return bar_final

          bar_with_first = neg_to_non_neg(bar_with_first)
          bar_with_last = neg_to_non_neg(bar_with_last)

          bar_with_first = tf.sign(bar_with_first)
          bar_with_last = tf.sign(bar_with_last)
          euclidean = tf.sqrt(tf.to_float(tf.reduce_sum(tf.square(bar_with_first - bar_with_last))))
          return euclidean

      if mode == 'train':
        a = 0.01
        extra_loss_result = a * extra_loss()
        loss = tf.reduce_mean(softmax_cross_entropy) + extra_loss_result
        perplexity = tf.exp(loss)
        accuracy = tf.reduce_mean(correct_predictions)
        event_accuracy = (
            tf.reduce_sum(correct_predictions * event_positions) /
            tf.reduce_sum(event_positions))
        no_event_accuracy = (
            tf.reduce_sum(correct_predictions * no_event_positions) /
            tf.reduce_sum(no_event_positions))

        loss_per_step = tf.reduce_sum(softmax_cross_entropy) / num_steps
        perplexity_per_step = tf.exp(loss_per_step)

        optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparams.clip_norm)
        tf.add_to_collection('train_op', train_op)

        vars_to_summarize = {
            'loss': loss,
            'metrics/perplexity': perplexity,
            'metrics/accuracy': accuracy,
            'metrics/event_accuracy': event_accuracy,
            'metrics/no_event_accuracy': no_event_accuracy,
            'metrics/loss_per_step': loss_per_step,
            'metrics/perplexity_per_step': perplexity_per_step,
            'metrics/extra_loss': extra_loss_result,
        }
      elif mode == 'eval':
        vars_to_summarize, update_ops = tf.contrib.metrics.aggregate_metric_map(
            {
                'loss': tf.metrics.mean(softmax_cross_entropy),
                'metrics/accuracy': tf.metrics.accuracy(
                    labels_flat, predictions_flat),
                'metrics/per_class_accuracy':
                    tf.metrics.mean_per_class_accuracy(
                        labels_flat, predictions_flat, num_classes),
                'metrics/event_accuracy': tf.metrics.recall(
                    event_positions, correct_predictions),
                'metrics/no_event_accuracy': tf.metrics.recall(
                    no_event_positions, correct_predictions),
                'metrics/loss_per_step': tf.metrics.mean(
                    tf.reduce_sum(softmax_cross_entropy) / num_steps,
                    weights=num_steps),
            })
        for updates_op in update_ops.values():
          tf.add_to_collection('eval_ops', updates_op)

        # Perplexity is just exp(loss) and doesn't need its own update op.
        vars_to_summarize['metrics/perplexity'] = tf.exp(
            vars_to_summarize['loss'])
        vars_to_summarize['metrics/perplexity_per_step'] = tf.exp(
            vars_to_summarize['metrics/loss_per_step'])

      for var_name, var_value in six.iteritems(vars_to_summarize):
        tf.summary.scalar(var_name, var_value)
        tf.add_to_collection(var_name, var_value)

    elif mode == 'generate':
      temperature = tf.placeholder(tf.float32, [])
      softmax_flat = tf.nn.softmax(
          tf.div(logits_flat, tf.fill([num_classes], temperature)))
      softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('temperature', temperature)
      tf.add_to_collection('softmax', softmax)
      # Flatten state tuples for metagraph compatibility.
      for state in tf_nest.flatten(initial_state):
        tf.add_to_collection('initial_state', state)
      for state in tf_nest.flatten(final_state):
        tf.add_to_collection('final_state', state)

  return build
