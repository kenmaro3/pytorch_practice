from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from model import *


if __name__ == "__main__":
    char2id = {str(i): i for i in range(10)}

    char2id.update({" ": 10, "-": 11, "_": 12})

    def generate_number():
        number = [random.choice(list("0123456789")) for _ in range(random.randint(1,3))]
        return int("".join(number))


    def add_padding(number, is_input=True):
        number = "{: <7}".format(number) if is_input else "{: <5s}".format(number)
        return number

    num = generate_number()
    print("\"" + str(add_padding(num)) + "\"")

    input_data = []
    output_data = []

    while len(input_data) < 5000:
        x = generate_number()
        y = generate_number()
        z = x - y
        input_char = add_padding(str(x) + "-" + str(y))
        output_char = add_padding("_" + str(z), is_input=False)

        input_data.append([char2id[c] for c in input_char])
        output_data.append([char2id[c] for c in output_char])


    print(input_data[100])
    print(output_data[100])

    train_x, test_x, train_y, test_y = train_test_split(input_data, output_data, train_size=0.7)


    def train2batch(input_data, output_data, batch_size=100):
        input_batch, output_batch = [], []
        input_shuffle, output_shuffle = shuffle(input_data, output_data)

        for i in range(0, len(input_data), batch_size):
            input_batch.append(input_shuffle[i:i+batch_size])
            output_batch.append(output_shuffle[i:i+batch_size])
        return input_batch, output_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, char2id).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, char2id).to(device)

    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.01)


    BATCH_NUM = 100
    EPOCH_NUM = 100

    all_losses = []
    print("training...")
    for epoch in range(1, EPOCH_NUM+1):
        epoch_loss = 0

        input_batch, output_batch = train2batch(train_x, train_y, batch_size=BATCH_NUM)

        for i in range(len(input_batch)):

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()


            input_tensor = torch.zeros(len(input_batch), len(input_batch[i][0]), device=device).long()
            output_tensor = torch.zeros(len(output_batch), len(output_batch[i][0]), device=device).long()
            for j in range(len(input_batch)):
                input_tensor[j, :] = torch.tensor(input_batch[i][j])
                output_tensor[j, :] = torch.tensor(output_batch[i][j])

            encoder_state = encoder(input_tensor)

            source = output_tensor[:, :-1]

            target = output_tensor[:, 1:]

            loss = 0

            decoder_output, _ = decoder(source, encoder_state)
            # decoder_output.size() = (100, 4, 13)


            for j in range(decoder_output.size()[1]):
                loss += criterion(decoder_output[:, j, :], target[:, j])

            epoch_loss += loss.item()

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

        print("Epoch %d: %.2f" % (epoch, epoch_loss))
        all_losses.append(epoch_loss)
        if epoch_loss < 1: break

    print("done.")


    #plt.plot(all_losses)
    #plt.show()

