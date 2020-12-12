from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim

input_date = []
output_date = []

test_size = 5000

file_path = "./mydate.txt"

with open(file_path, "r") as f:
    date_list = f.readlines()
    for i, date in enumerate(date_list):
        date = date[:-1]
        input_date.append(date.split("_")[0])
        output_date.append("_" + date.split("_")[1])
        if i == test_size: break


input_len = len(input_date[0])
output_len = len(output_date[0])

char2id = {}
for input_chars, output_chars in zip(input_date, output_date):
    for c in input_chars:
        if not c in char2id:
            char2id[c] = len(char2id)
    for c in output_chars:
        if not c in char2id:
            char2id[c] = len(char2id)

input_data = []
output_data = []

for input_chars, output_chars in zip(input_date, output_date):
    input_data.append([char2id[c] for c in input_chars])
    output_data.append([char2id[c] for c in output_chars])

train_x, test_x, train_y, test_y = train_test_split(input_data, output_data, train_size=0.7)

print(len(train_x))
print(len(test_x))

embedding_dim = 200
hidden_dim = 128
BATCH_NUM=100
EPOCH_NUM=15
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

        for i in range(attention_weight.size()[2]): # 10 iteration
            unsq_weight = attention_weight[:,:,i].unsqueeze(2) # unsq_weight.size() = 100, 29, 1

            weighted_hs = hs * unsq_weight # weighted_hs.size() = 100, 29, 128

            weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(1) # weight_sum.size() = 100, 1, 128

            c = torch.cat([c, weight_sum], dim=1) # c.size() = 100, i, 128

        c = c[:, 1:, :]

        output = torch.cat([output, c], dim=2) # output.size() = 100, 10, 256
        output = self.hidden2linear(output)
        return output, state, attention_weight




if __name__ == "__main__":

    def train2batch(input_data, output_data, batch_size=100):
        input_batch, output_batch = [], []
        input_shuffle, output_shuffle = shuffle(input_data, output_data)
        for i in range(0, len(input_data), batch_size):
            input_batch.append(input_shuffle[i:i+batch_size])
            output_batch.append(output_shuffle[i:i+batch_size])
        return input_batch, output_batch

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
    attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM).to(device)

    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01)
    attn_decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=0.01)


    all_losses = []
    print("training ...")
    for epoch in range(1, EPOCH_NUM+1):
        epoch_loss = 0

        input_batch, output_batch = train2batch(train_x, train_y, batch_size=BATCH_NUM)

        for i in range(len(input_batch)):
            encoder_optimizer.zero_grad()
            attn_decoder_optimizer.zero_grad()

            input_tensor = torch.zeros(len(input_batch[i]), len(input_batch[i][0])).long()
            output_tensor = torch.zeros(len(output_batch[i]), len(output_batch[i][0])).long()
            for j in range(len(input_batch[i])):
                input_tensor[j] = torch.LongTensor(input_batch[i][j])
                output_tensor[j] = torch.LongTensor(output_batch[i][j])

            hs, h = encoder(input_tensor)

            source = output_tensor[:, :-1]

            target = output_tensor[:, 1:]

            loss = 0
            decoder_output, _, attention_weight = attn_decoder(source, hs, h)

            for j in range(decoder_output.size()[1]):
                loss += criterion(decoder_output[:, j, :], target[:, j])

            epoch_loss += loss.item()

            loss.backward()

            encoder_optimizer.step()
            attn_decoder_optimizer.step()

        print("epoch %d: %.2f" % (epoch, epoch_loss))
        all_losses.append(epoch_loss)
        if epoch_loss < 0.1: break

    print("done")







