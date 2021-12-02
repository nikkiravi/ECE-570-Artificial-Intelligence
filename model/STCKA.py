import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math


class STCKAtten(nn.Module):
	def __init__(self, train_vocab_length, concepts_vocab_length, embedding_dimension, train_embedding_weights, concept_embedding_weights, hidden_dimension, output_dimension, network = "cnn", gamma=0.5, number_layers = 1):
		super(STCKAtten, self).__init__()

		self.gamma = gamma # soft switch to adjust the importance of two attention weights, alpha_i and beta_i
		da = hidden_dimension # the hyperparameter for the weight of attention alpha_i
		db = int(da / 2) # the hyperparameter for the weight of attention beta_i

		self.train_word_embeddings = nn.Embedding(train_vocab_length, embedding_dimension) # lookup table that stores embeddings of a fixed dictionary
		if(isinstance(train_embedding_weights, torch.Tensor)):
			self.train_word_embeddings.weight = nn.Parameter(train_embedding_weights, requires_grad = True) # add the train embedding weights parameters to the module parameters: parameters() iterator

		self.concept_word_embeddings = nn.Embedding(concepts_vocab_length, embedding_dimension)
		if(isinstance(concept_embedding_weights, torch.Tensor)):
			self.concept_word_embeddings.weight = nn.Parameter(concept_embedding_weights, requires_grad = True) # add the concept embedding weights parameters to the module parameters: parameters() iterator

		self.lstm = nn.LSTM(embedding_dimension, hidden_dimension, num_layers = number_layers, batch_first = True, bidirectional = True) # long short-term memory (LSTM) RNN for short-text encoding

		if(network == "cnn"):
			self.alpha_W1, self.alpha_w1, self.beta_W2, self.beta_w2, self.output = self.convolutional_network(hidden_dimension, embedding_dimension, da, db, output_dimension)
		else:
			self.alpha_W1, self.alpha_w1, self.beta_W2, self.beta_w2, self.output = self.linear_network(hidden_dimension, embedding_dimension, da, db, output_dimension)


	def convolutional_network(self, hidden_dimension, embedding_dimension, da, db, output_dimension):
		print("Implementing Convolutional Neural Network STCKA Model")
		alpha_W1 = nn.Conv1d(in_channels = 2 * hidden_dimension + embedding_dimension, out_channels = da, kernel_size = 3, stride = 1, padding = 1)
		alpha_w1 = nn.Conv1d(in_channels = da, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
		beta_W2 = nn.Conv1d(in_channels = embedding_dimension, out_channels = db, kernel_size = 3, stride = 1, padding = 1)
		beta_w2 = nn.Conv1d(in_channels = db, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
		output = nn.Conv1d(in_channels = 2 * hidden_dimension + embedding_dimension, out_channels = output_dimension, kernel_size = 3, stride = 1, padding = 1)
		
		return alpha_W1, alpha_w1, beta_W2, beta_w2, output

	def linear_network(self, hidden_dimension, embedding_dimension, da, db, output_dimension):
		print("Implementing Fully Connected Linear STCKA Model")
		alpha_W1 = nn.Linear(in_features = 2 * hidden_dimension + embedding_dimension, out_features = da)
		alpha_w1 = nn.Linear(in_features = da, out_features = 1, bias = False) # The layer will not learn additive bias seeing as both alpha weights are supposed to be computed in one line
		beta_W2 = nn.Linear(in_features = embedding_dimension, out_features = db)
		beta_w2 = nn.Linear(in_features = db, out_features = 1, bias = False) # The layer will not learn additive bias seeing as both alpha weights are supposed to be computed in one line
		output = nn.Linear(in_features = 2 * hidden_dimension + embedding_dimension, out_features = output_dimension)

		return alpha_W1, alpha_w1, beta_W2, beta_w2, output

	def self_attention(self, H):
		# H = Output of the Bidirectional LSTM (batch_size, input_seq_length, 2 * hidden_dimension)
		# Implementing A = Attention(Q,K,V) = softmax((QK^T) / (sqrt(2u)))V, u = hidden_dimension
		Q = K = V = H
		u = H.size()[-1] / 2 # hidden_dimension
		attention = F.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(2*u), -1) # dimension: batch_size, input_seq_length, input_seq_length
		attention = torch.bmm(attention, V).permute(0, 2, 1) # dimension: batch_size, 2 * hidden_dimension, input_seq_length

		q = F.max_pool1d(attention, attention.size()[2]).squeeze(-1) # Acquire short-text representation, dimension: batch_size, 2 * hidden_dimension
		return q

	def cst_attention(self, c, q, network="cnn"):
		# c is the ith concept vector
		q = q.unsqueeze(1) # dimension (batch_size, 1, 2 * hidden size)
		q = q.expand(q.size()[0], c.size()[1], q.size()[2]) # expanding to make q the same dimension as concept vector
		cq = torch.cat((c, q), -1) # dimensions are: batch_size, concept_seq_length, hidden_dimension * 2 + embedding_dimension
		if(network == "cnn"):
			cq = cq.permute(0,2,1) # dimensions are: batch_size, hidden_dimension * 2 + embedding_dimension, concept_seq_length

		alpha = self.alpha_w1(torch.tanh(self.alpha_W1(cq))) # dimensions are: batch_size, 1, concept_seq_length
		if(network == "cnn"):
			alpha = alpha.permute(0,2,1) # dimensions are: batch_size, concept_seq_length, 1

		alpha = F.softmax(alpha.squeeze(-1), -1) # dimensions are: batch_size, concept_seq_length
		
		return alpha

	def ccs_attention(self, c, network="cnn"):
		# c is the ith concept vector
		if(network == "cnn"):
			c = c.permute(0,2,1) # dimensions are: batch_size, embedding_dim, concept_seq_len

		beta = self.beta_w2(torch.tanh(self.beta_W2(c))) # dimensions are: batch_size, 1, concept_seq_length
		if(network == "cnn"):
			beta = beta.permute(0,2,1) # dimensions are: batch_size, concept_seq_length, 1

		beta = F.softmax(beta.squeeze(-1), -1) # dimensions are: batch_size, concept_seq_length

		return beta

	def forward(self, short_text, concepts, network="cnn"):
		# short_text dimension: batch_size, text_seq_length
		# concepts dimension: batch_size, concept_seq_length
		input_short_text = self.train_word_embeddings(short_text) # text embedding step ->  batch_size, text_seq_length, embedding_dimension
		output, (h_n, c_n) = self.lstm(input_short_text) # LSTM requires both the batch and sequence length which is why include_lengths = True in data.Field, output dimension is: (batch_size, input_seq_length, 2 * hidden_dimension)
		q = self.self_attention(output) # short text representation

		input_concepts = self.concept_word_embeddings(concepts) # concept embedding step -> batch_size, concept_seq_length, embedding_dimension
		alpha = self.cst_attention(input_concepts, q, network) # dimensions are: batch_size, concept_seq_length
		beta = self.ccs_attention(input_concepts, network) # dimensions are: batch_size, concept_seq_length

		a_i = F.softmax((self.gamma * alpha) + (1 - self.gamma) * beta, -1) # Dimensions are: batch_size, concept_seq_length
		p = torch.bmm(a_i.unsqueeze(1), input_concepts).squeeze(1) # batch_size, embedding_dimension

		final = torch.cat((q, p), -1) # dimensions are: batch_size, 2 * hidden_dimension + embedding_dimension
		if(network == "cnn"):
			final = final.unsqueeze(-1)

		output = self.output(final)
		output = output.squeeze() # dimensions are: batch_size, output_dimension
		return output

