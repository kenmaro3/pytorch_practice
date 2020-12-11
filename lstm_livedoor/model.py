import torch
import torch.nn as nn
import torch.optim as optim


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMClassifier, self).__init__()

        # dimension of hidden layer
        self.hidden_dim = hidden_dim

        # dimension of word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # nn.LSTM(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # hidden dimension to target_size dimension mapping
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.view(embeds.size()[0], 1, embeds.size()[1])
        _, (lstm_out_h, lstm_out_c) = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out_h.view(-1, self.hidden_dim))
        tag_scores = self.softmax(tag_space)
        return tag_scores






