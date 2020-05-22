# modified version of https://github.com/datalogue/keras-attention/
import tensorflow as tf
import keras
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.layers import Embedding
from keras.engine import InputSpec


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # maybe below is more clear implementation compared to older keras
    # at least it works the same for tensorflow, but not tested on other backends
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    return x


class AttentionDecoder(Recurrent):

    def __init__(self, units, alphabet_size,
                 embedding_dim=30,
                 is_monotonic=False,
                 normalize_energy=False,
                 activation='tanh',
                 dropout=None,
                 recurrent_dropout=None,
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param alphabet_size: output sequence alphabet size
            (alphabet may contain <end_of_seq> but do not need to have <start_of_seq>,
            because it is added internally inside the layer)
        :param embedding_dim: size of internal embedding for output labels
        :param is_monotonic: if True - Luong-style monotonic attention
            if False - Bahdanau-style attention (non-monotonic)
            See references for details
        :param normalize_energy: whether attention weights are normalized
        references:
            (1) Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).

            (2) Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglass Eck
            "Online and Linear-Time Attention by Enforcing Monotonic Alignments"
            arXiv arXiv:1704.00784 (2017)
        notes:
            with `is_monotonic=True`, `normalize_energy=True` equal to model in (2)
            with `is_monotonic=False`, `normalize_energy=False` equal to model in (1)
        """
        self.start_token = alphabet_size
        output_dim = alphabet_size  # alphabet + end_token

        self.units = units
        self.alphabet_size = alphabet_size
        self.output_dim = output_dim

        self.is_monotonic = is_monotonic
        self.normalize_energy = normalize_energy

        self.embedding_dim = embedding_dim
        self.embedding_sublayer = Embedding(alphabet_size + 1, embedding_dim)
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences
        self.return_state = False
        self.stateful = False
        self.uses_learning_phase = True

    def add_scalar(self, initial_value=0, name=None, trainable=True):
        scalar = K.variable(initial_value, name=name)
        if trainable:
            self._trainable_weights.append(scalar)
        else:
            self._non_trainable_weights.append(scalar)
        return scalar

    def build(self, input_shapes):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.

          See Luong 2017, arXiv:1704.00784
          for model details that correspond to the scalars here.
        """
        input_shape = input_shapes
        if isinstance(input_shapes[0], (list, tuple)):
            if len(input_shapes) > 2:
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 '1 or 2 input tensors, but it received ' +
                                 str(len(input_shapes)) + ' input tensors.')

            self.input_spec = [InputSpec(shape=input_shape) for input_shape in input_shapes]
            input_shape = input_shapes[0]
        else:
            self.input_spec = [InputSpec(shape=input_shape)]
        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None, None]  # y, s, t

        """
            Embedding matrix for y (outputs)
        """
        self.embedding_sublayer.build(input_shape=(self.batch_size, self.input_dim))

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.embedding_dim, self.units),
                                   name='W_r',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_r = self.add_weight(shape=(self.units,),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.embedding_dim, self.units),
                                   name='W_z',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_z = self.add_weight(shape=(self.units,),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.embedding_dim, self.units),
                                   name='W_p',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_p = self.add_weight(shape=(self.units,),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.embedding_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim,),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        if self.is_monotonic:
            self.Energy_r = self.add_scalar(initial_value=-1,
                                            name='r')
            self.states.append(None)
        if self.normalize_energy:
            self.Energy_g = self.add_scalar(initial_value=1,
                                            name='g')

        self.built = True

    def call(self, x, use_teacher_forcing=True, training=None):
        # TODO: check that model is loading from .h5 correctly
        # TODO: for now cannot be shared layer
        # (only can it we use (or not use) teacher forcing in all cases simultationsly)

        # this sequence is used only to extract the amount of timesteps (the same as in output sequence)
        fake_input = x
        if isinstance(x, list):
            # teacher forcing for training
            self.x_seq, self.y_true = x
            self.use_teacher_forcing = use_teacher_forcing
            fake_input = K.expand_dims(self.y_true)
        else:
            # inference
            self.x_seq = x
            self.use_teacher_forcing = False

        # apply a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # therefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             dropout=self.dropout,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units,
                                             training=training)

        last_output, outputs, states = K.rnn(
            self.step,
            inputs=fake_input,
            initial_states=self.get_initial_state(self.x_seq)
        )
        return outputs

    def get_initial_state(self, inputs):
        if isinstance(inputs, list):
            assert len(inputs) == 2  # inputs == [encoder_outputs, y_true]
            encoder_outputs = inputs[0]
        else:
            encoder_outputs = inputs

        memory_shape = K.shape(encoder_outputs)

        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(encoder_outputs[:, 0], self.W_s))

        y0 = K.zeros((memory_shape[0],), dtype='int64') + self.start_token
        t0 = K.zeros((memory_shape[0],), dtype='int64')

        initial_states = [y0, s0, t0]
        if self.is_monotonic:
            # initial attention has form: [1, 0, 0, ..., 0] for each sample in batch
            alpha0 = K.ones((memory_shape[0], 1))
            alpha0 = K.switch(K.greater(memory_shape[1], 1),
                              lambda: K.concatenate([alpha0, K.zeros((memory_shape[0], memory_shape[1] - 1))], axis=-1),
                              alpha0)
            # like energy, attention is stored in shape (samples, time, 1)
            alpha0 = K.expand_dims(alpha0, -1)
            initial_states.append(alpha0)

        return initial_states

    def step(self, x, states):
        if self.is_monotonic:
            ytm, stm, timestep, previous_attention = states
        else:
            ytm, stm, timestep = states

        ytm = self.embedding_sublayer(K.cast(ytm, 'int32'))

        if self.recurrent_dropout is not None and 0. < self.recurrent_dropout < 1.:
            stm = K.in_train_phase(K.dropout(stm, self.recurrent_dropout), stm)
            ytm = K.in_train_phase(K.dropout(ytm, self.recurrent_dropout), ytm)

        et = self._compute_energy(stm)

        if self.is_monotonic:
            at = self._compute_probabilities(et, previous_attention)
        else:
            at = self._compute_probabilities(et)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)

        # ~~~> calculate new hidden state

        # first calculate the "r" gate:
        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1 - zt) * stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(st, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.use_teacher_forcing:
            ys = K.in_train_phase(self.y_true[:, timestep[0]], K.argmax(yt, axis=-1))
            ys = K.flatten(ys)
        else:
            ys = K.flatten(K.argmax(yt, axis=-1))

        if self.return_probabilities:
            output = at
        else:
            output = yt

        next_states = [ys, st, timestep + 1]
        if self.is_monotonic:
            next_states.append(at)
        return output, next_states

    def _compute_energy(self, stm):
        # "concat" energy function
        # energy_i = g * V / |V| * tanh([stm, h_i] * W + b) + r
        _stm = K.dot(stm, self.W_a)

        V_a = self.V_a
        if self.normalize_energy:
            V_a = self.Energy_g * K.l2_normalize(self.V_a)

        et = K.dot(activations.tanh(K.expand_dims(_stm, axis=1) + self._uxpb),
                   K.expand_dims(V_a))

        if self.is_monotonic:
            et += self.Energy_r

        return et

    def _compute_probabilities(self, energy, previous_attention=None):
        if self.is_monotonic:
            # add presigmoid noise to encourage discreteness
            sigmoid_noise = K.in_train_phase(1., 0.)
            noise = K.random_normal(K.shape(energy), mean=0.0, stddev=sigmoid_noise)
            # encourage discreteness in train
            energy = K.in_train_phase(energy + noise, energy)

            p = K.in_train_phase(K.sigmoid(energy),
                                 K.cast(energy > 0, energy.dtype))
            p = K.squeeze(p, -1)
            p_prev = K.squeeze(previous_attention, -1)
            # monotonic attention function from tensorflow
            at = K.in_train_phase(
                tf.contrib.seq2seq.monotonic_attention(p, p_prev, 'parallel'),
                tf.contrib.seq2seq.monotonic_attention(p, p_prev, 'hard'))
            at = K.expand_dims(at, -1)
        else:
            # softmax
            at = keras.activations.softmax(energy, axis=1)

        return at

    def compute_output_shape(self, input_shapes):
        """
            For Keras internal compatability checking
        """
        input_shape = input_shapes
        if isinstance(input_shapes[0], (list, tuple)):
            input_shape = input_shapes[0]

        timesteps = input_shape[1]
        if self.return_probabilities:
            return (None, timesteps, timesteps)
        else:
            return (None, timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'units': self.units,
            'alphabet_size': self.alphabet_size,
            'embedding_dim': self.embedding_dim,
            'return_probabilities': self.return_probabilities,
            'is_monotonic': self.is_monotonic,
            'normalize_energy': self.normalize_energy
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
