import torch

import torch.nn as nn
import torch.optim as optim

embedding_dim = 200
hidden_dim = 128
batch_num = 100
vocab_size = len(char2id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 29 is seq_len for input to encoder
# 10 is seq_len for output from decoder

class Encoder(nn.Module):
    # encoder which is many to many
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id[" "])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        # sequence.size() = (batch, seq_len, word_idx)
        # embedding.size() = (batch, seq_len, embedding_dim)
        embedding = self.word_embeddings(sequence)

        # hs.size() = (batch, seq_len, hidden_size*num_directions)
        # h.size() =  (batch, num_layers*num_directions, hidden_size)
        hs, h = self.gru(embedding)
        return hs, h


class AttentionDecoder(nn.Module):
    # 
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id[" "])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2linear = nn.Linear(hidden_dim * 2, vocab_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, hs, h):
        # sequence.size() = (batch, seq_len_out, word_idx) # seq_len_out == 10
        # hs.size() = (batch, seq_len_in, hidden_size*num_directions) # seq_len_in = 29
        # h.size() = (batch, num_layers*num_directions, hidden_size)

        # embedding.size() = (batch, seq_len_out, embedding_size)
        embedding = self.word_embeddings(sequence)

        # output.size() = (batch, seq_len_out, hidden_size*num_directions)
        # state.size() = (batch, num_layers*num_directions, hidden_size)
        output, state = self.gru(embedding, h)

        # output.size() = (100, 10, 128) # batch, seq_len_out, embedding_size
        # hs.size() = (100, 29, 128) # batch, seq_len_in, embedding_size

        # t_output.size() = (batch, hidden_size, seq_len_out)
        t_output = torch.transpose(output, 1, 2) # t_output.size() = (100, 128, 120)
        
        # s.size() = torch.bmm((batch, seq_len_in, hidden_size), (batch, hidden_size, seq_len_out))
        # s.size() = (batch, seq_len_in, seq_len_out)
        s = torch.bmm(hs, t_output) # s.size() = 100, 29, 10
        
        # taking softmax along seq_len_in ==> means which input sequence has the most similarity
        # attention_weight.size() = (batch, seq_len_in, seq_len_out)
        attention_weight = self.softmax(s) # attention_weight.size() = 100, 29, 10
        
        # c.size() = (batch, 1, hidden_dim)
        c = torch.zeros(self.batch_size, 1, self.hidden_dim, device=device) # c.size() = 10, 1, 128
        
        # loop over seq_len_out
        for i in range(atetntion_weight.size()[2]): # 10 iteration
            unsq_weight = attention_weight[:,:,i].unsqueeze(2) # unsq_weight.size() = 100, 29, 1
            
            # taking attention by inner product of
            # [(batch, seq_len_in, hidden_size), (batch, seq_len_in, 1)]
            # weighted_hs.size() = (batch, seq_len_in, hidden_size)
            weighted_hs = hs * unsq_weight # weighted_hs.size() = 100, 29, 128
            
            # weight_sum.size() = (batch, 1, hidden_size)
            weight_sum = torch.sum(weight_hs, axis=1).unsqueeze(1) # weight_sum.size() = 100, 1, 128
            # c.size() = (batch, 1+i, hidden_size)
            c = torch.cat([c, weight_sum], dim=1) # c.size() = 100, i, 128
        
        # c.size() = (batch, 1+seq_len_out, hidden_size)
        # c.size() = (batch, seq_len_out, hidden_size) 
        c = c[:, 1:, :]
        
        # output.size() = (batch, seq_len_out, hidden_size*2) 
        output = torch.cat([output, c], dim=2) # output.size() = 100, 10, 256

        # output.size() = (batch, seq_len_out, vocab_size)
        output = self.hidden2linear(output)
        
        # attention_weight.size() = (batch, seq_len_in, seq_len_out)
        return output, state, attention_weight



