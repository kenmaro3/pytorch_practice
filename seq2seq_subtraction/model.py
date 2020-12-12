import torch
import torch.nn as nn
import torch.optim as optim

embedding_dim  = 200
hidden_dim  = 128
vocab_size = 13


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, char2idx):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2idx[" "])
        self.lstm  = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)

        _, state = self.lstm(embedding)
        # state = (h, c)
        return state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, char2idx):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2idx[" "])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sequence, encoder_state):
        embedding = self.word_embeddings(sequence)

        output, state = self.lstm(embedding, encoder_state)
        output = self.hidden2linear(output)
        return output, state



