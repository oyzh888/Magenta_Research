# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for chord_inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.music import chord_inference
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2


class ChordInferenceTest(tf.test.TestCase):

  def testSequenceNotePitchVectors(self):
    sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.0, 0.0), (62, 100, 0.0, 0.5),
         (60, 100, 1.5, 2.5),
         (64, 100, 2.0, 2.5), (67, 100, 2.25, 2.75), (70, 100, 2.5, 4.5),
         (60, 100, 6.0, 6.0),
        ])
    note_pitch_vectors = chord_inference.sequence_note_pitch_vectors(
        sequence, seconds_per_frame=1.0)

    expected_note_pitch_vectors = [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    self.assertEqual(expected_note_pitch_vectors, note_pitch_vectors.tolist())

  def testInferChordsForSequence(self):
    sequence = music_pb2.NoteSequence()
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(60, 100, 0.0, 1.0), (64, 100, 0.0, 1.0), (67, 100, 0.0, 1.0),   # C
         (62, 100, 1.0, 2.0), (65, 100, 1.0, 2.0), (69, 100, 1.0, 2.0),   # Dm
         (60, 100, 2.0, 3.0), (65, 100, 2.0, 3.0), (69, 100, 2.0, 3.0),   # F
         (59, 100, 3.0, 4.0), (62, 100, 3.0, 4.0), (67, 100, 3.0, 4.0)])  # G
    quantized_sequence = sequences_lib.quantize_note_sequence(
        sequence, steps_per_quarter=4)
    chord_inference.infer_chords_for_sequence(
        quantized_sequence, chords_per_bar=2)

    expected_chords = [('C', 0.0), ('Dm', 1.0), ('F', 2.0), ('G', 3.0)]
    chords = [(ta.text, ta.time) for ta in quantized_sequence.text_annotations]

    self.assertEqual(expected_chords, chords)


if __name__ == '__main__':
  tf.test.main()
