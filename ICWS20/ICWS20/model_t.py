import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import os

# Device configuration
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cpu"


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    data = np.load(name, allow_pickle=True)  # 使用numpy加载.npy文件
    data = data[0:int(len(data)/5)]
    count = 0
    word_list = []
    for group in tqdm(data, total=len(data), desc="Loading"):
        sentence = []
        for line in group[0]:
            num_sessions += 1
            word = line[1] + "_" + line[2]
            if word not in word_list:
                word_list.append(word)
            sentence.append(word_list.index(word))
        label = group[1]

        if label == 1:
            count += 1

        scaler = StandardScaler()
        sentence = scaler.fit_transform(np.array(sentence).reshape(-1, 1))
        sentence = sentence.reshape(-1)
        sentence = sentence.tolist()

        inputs.append(sentence)
        outputs.append(label)

    print("Length of word_list({}): {}".format(name, len(word_list)))
    print("Number of sessions({}): {}".format(name, num_sessions))
    print("Number of seqs({}): {}".format(name, len(inputs)))
    print("Number of anomalies({}): {}".format(name, count))

    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs)
    )

    return dataset


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1])
        return out


if __name__ == "__main__":
    # Hyperparameters
    num_classes = 2
    num_epochs = 100
    batch_size = 512
    input_size = 1000
    vocab_size = 384
    learning_rate = 0.001
    model_dir = "ICWS20/ICWS20/model"
    log = "Adam_batch_size={}_epoch={}".format(str(batch_size), str(num_epochs))
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_layers", default=3, type=int)
    parser.add_argument("-hidden_size", default=64, type=int)
    parser.add_argument("-window_size", default=100, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    start_time = time.time()
    for fn in os.listdir("ICWS20/Drain3/data/platform"):

        model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
        seq_dataset = generate(f"ICWS20/Drain3/data/platform/{fn}")

        total_size = len(seq_dataset)
        train_size = int(0.8 * total_size)  # 80% for training, 20% for testing
        test_size = total_size - train_size
        train_dataset, test_dataset = random_split(seq_dataset, [train_size, test_size])

        # DataLoader for training and testing sets
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        data_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True)
        # print(seq_dataset)
        # writer = SummaryWriter(log_dir="ICWS20/DeepLog/log/" + log)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        total_step = len(data_loader)
        for epoch in range(num_epochs):  # Loop over the dataset multiple times
            model.train()
            train_loss = 0
            for step, (seq, label) in enumerate(train_dataloader):
                # Forward pass
                # seq = seq.clone().detach().view(window_size, -1, input_size).to(device)
                # print(seq.shape)
                output = model(seq)
                # print(output.shape, label.shape)
                loss = criterion(output, label.to(device))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                # writer.add_graph(model, seq)
                # print(output, label)
            print(
                "Epoch [{}/{}], train_loss: {:.4f}".format(
                    epoch + 1, num_epochs, train_loss / total_step
                )
            )
            # writer.add_scalar("train_loss", train_loss / total_step, epoch + 1)

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), model_dir + "/" + log + ".pt")
        # writer.close()

        anormaly_score = []
        for step, (seq, label) in enumerate(data_loader):
            output = model(seq)
            anormaly_score.append(output.detach().numpy())

        anormaly_score = np.vstack(anormaly_score)
        print(anormaly_score.shape)
        np.save(f"ICWS20/ICWS20/output/{fn.split('_')[0]}_ts.npy", anormaly_score)
    elapsed_time = time.time() - start_time
    print("elapsed_time: {:.3f}s".format(elapsed_time))