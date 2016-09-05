import tensorflow as tf
import numpy as np


class TextRNN(object):
	"""
	A RNN for text classification.
	Uses an RNN with two-layered LSTM units followed by an average pooling
	"""
	def __init__(
	  self, max_seq_length, num_classes, vocab_size,
	  embedding_size, num_lstm_layers, l2_reg_lambda=0.0):

		# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.int32, [None, max_seq_length], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob_lstm_input = tf.placeholder(tf.float32, name="dropout_keep_prob_input")
		self.dropout_keep_prob_lstm_output = tf.placeholder(tf.float32, name="dropout_keep_prob_output")
		self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
		self.input_lengths = tf.placeholder(tf.int32, [None], name="input_lengths")

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			self.w_embedding = tf.Variable(
				tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
				name="W")
			self.embedded_chars = tf.nn.embedding_lookup(self.w_embedding, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		with tf.variable_scope("lstm") as scope:
			# The RNN cell
			single_cell = tf.nn.rnn_cell.DropoutWrapper(
				tf.nn.rnn_cell.LSTMCell(embedding_size, embedding_size, initializer=tf.random_uniform_initializer(-1.0, 1.0)),
				input_keep_prob=self.dropout_keep_prob_lstm_input,
				output_keep_prob=self.dropout_keep_prob_lstm_output)
			self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_lstm_layers)
			# Build the recurrence. We do this manually to use truncated backprop
			self.initial_state = tf.zeros([self.batch_size, self.cell.state_size])
			self.encoder_states = [self.initial_state]
			self.encoder_outputs = []

			# Truncate the input lengths
			self.input_lengths = tf.minimum(self.input_lengths, max_seq_length)

			# RNN loop
			for i in range(max_seq_length):
				if i > 0:
					scope.reuse_variables()
				new_output, new_state = self.cell(self.embedded_chars[:, i, :], self.encoder_states[-1])
				# create mask to invalidate states that exceed a sentence's length
				step_matrix = tf.zeros([self.batch_size, self.cell.state_size]) + i
				mask = tf.transpose(tf.less(tf.transpose(step_matrix), tf.to_float(self.input_lengths)))
				valid_state = tf.select(mask, new_state, self.initial_state)
				self.encoder_outputs.append(new_output)
				self.encoder_states.append(valid_state)

			concat_states =  tf.pack(self.encoder_states)

			# Compute average based on each sentence's length
			avg_states = tf.transpose(tf.transpose(tf.reduce_sum(concat_states, 0)) / tf.to_float(self.input_lengths))

			_, self.final_state = tf.split(1,2,avg_states)
			self.final_state = tf.slice(self.final_state, [0,embedding_size*(num_lstm_layers - 1)], [-1,embedding_size])
			#self.final_output = self.encoder_outputs[-1]

		with tf.variable_scope("output_projection"):
			W = tf.get_variable(
				"W",
				[embedding_size, num_classes],
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			b = tf.get_variable(
				"b",
				[num_classes],
				initializer=tf.constant_initializer(0.1))
			self.scores = tf.nn.xw_plus_b(self.final_state, W, b)
			self.y = tf.nn.softmax(self.scores)
			self.predictions = tf.argmax(self.scores, 1)

		with tf.variable_scope("loss"):
			self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y, name="ce_losses")
			self.total_loss = tf.reduce_sum(self.losses)
			self.mean_loss = tf.reduce_mean(self.losses)

		with tf.variable_scope("accuracy"):
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
