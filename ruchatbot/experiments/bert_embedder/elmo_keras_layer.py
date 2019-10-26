""" https://github.com/JHart96/keras_elmo_embedding_layer """
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class ELMoEmbedding(Layer):

    def __init__(self, idx2word, output_mode="default", trainable=True, **kwargs):
        assert output_mode in ["default", "word_emb", "lstm_outputs1", "lstm_outputs2", "elmo"]
        assert trainable in [True, False]
        self.idx2word = idx2word
        self.output_mode = output_mode
        self.trainable = trainable
        self.max_length = None
        self.word_mapping = None
        self.lookup_table = None
        self.elmo_model = None
        self.embedding = None
        super(ELMoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.max_length = input_shape[1]
        self.word_mapping = [x[1] for x in sorted(self.idx2word.items(), key=lambda x: x[0])]
        self.lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(self.word_mapping, default_value="<UNK>")
        #self.lookup_table.init.run(session=K.get_session())

        #self.elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable)
        self.elmo_model = hub.Module("/mnt/7383b08e-ace3-49d3-8991-5b9aa07d2596/EmbeddingModels/deeppavlov_data/elmo_model", trainable=self.trainable)
        super(ELMoEmbedding, self).build(input_shape)

    def call(self, x):
        x = tf.cast(x, dtype=tf.int64)
        sequence_lengths = tf.cast(tf.count_nonzero(x, axis=1), dtype=tf.int32)
        strings = self.lookup_table.lookup(x)
        inputs = {
            "tokens": strings,
            "sequence_len": sequence_lengths
        }
        return self.elmo_model(inputs, signature="tokens", as_dict=True)[self.output_mode]

    def compute_output_shape(self, input_shape):
        if self.output_mode == "default":
            return (input_shape[0], 1024)
        if self.output_mode == "word_emb":
            return (input_shape[0], self.max_length, 512)
        if self.output_mode == "lstm_outputs1":
            return (input_shape[0], self.max_length, 1024)
        if self.output_mode == "lstm_outputs2":
            return (input_shape[0], self.max_length, 1024)
        if self.output_mode == "elmo":
            return (input_shape[0], self.max_length, 1024)

    def get_config(self):
        config = {
            'idx2word': self.idx2word,
            'output_mode': self.output_mode
        }
        return list(config.items())
