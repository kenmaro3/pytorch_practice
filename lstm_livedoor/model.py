import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMClassifier, self).__init__()

        # dimension of hidden layer
        self.hidden_dim = hidden_dim

        # dimension of word embeddings
        # padding_idx = 0 since <pad> has word id as 0
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # nn.LSTM(embedding_dim, hidden_dim)
        # batch_first = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # hidden dimension to target_size dimension mapping
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

        self.softmax = nn.LogSoftmax()


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

        # embeds.size() = (batch_size * len(sentence) * embedding_dim)
        #embeds = embeds.view(embeds.size()[0], 1, embeds.size()[1])
        _, (lstm_out_h, lstm_out_c) = self.lstm(embeds)
        #tag_space = self.hidden2tag(lstm_out_h.view(-1, self.hidden_dim))

        # lstm_out_h.size() = (1 * batch_size, hidden_dim)
        tag_space  = self.hidden2tag(lstm_out_h)

        # tag_space.size() = (1 * batch_size, tagset_size)
        # to make (batch_size * tagset_size), do squeeze()
        tag_scores = self.softmax(tag_space.squeeze())
        #tag_scores = self.softmax(tag_space)
        return tag_scores






