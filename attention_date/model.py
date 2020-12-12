import torch

import torch.nn as nn
import torch.optim as optim

embedding_dim = 200
hidden_dim = 128
batch_num = 100
vocab_size = len(char2id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id[" "])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        hs, h = self.gru(embedding)
        return hs, h


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id[" "])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2linear = nn.Linear(hidden_dim * 2, vocab_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, hs, h):
        embedding = self.word_embeddings(sequence)
        output, state = self.gru(embedding, h)

        # output.size() = (100, 10, 128) # batch, #, embedding_size
        # hs.size() = (100, 29, 128) # batch, #, embedding_size

        t_output = torch.transpose(output, 1, 2) # t_output.size() = (100, 128, 120)

        s = torch.bmm(hs, t_output) # s.size() = 100, 29, 10

        attention_weight = self.softmax(s) # attention_weight.size() = 100, 29, 10

        c = torch.zeros(self.batch_size, 1, self.hidden_dim, device=device) # c.size() = 10, 1, 128

        for i in range(atetntion_weight.size()[2]): # 10 iteration
            unsq_weight = attention_weight[:,:,i].unsqueeze(2) # unsq_weight.size() = 100, 29, 1

            weighted_hs = hs * unsq_weight # weighted_hs.size() = 100, 29, 128

            weight_sum = torch.sum(weight_hs, axis=1).unsqueeze(1) # weight_sum.size() = 100, 1, 128

            c = torch.cat([c, weight_sum], dim=1) # c.size() = 100, i, 128

        c = c[:, 1:, :]

        output = torch.cat([output, c], dim=2) # output.size() = 100, 10, 256
        output = self.hidden2linear(output)
        return output, state, attention_weight



