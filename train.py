import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import load_train
from model import CoronaryNet


LEARNING_RATE=5e-4
EPOCHS=300
BATCH_SIZE=2


def train_model(model, criterion, optimizer):
    _, dl_train = load_train(BATCH_SIZE)
    start = time.time()
    model.train()
    for epoch in range(EPOCHS):
        sum_loss = 0
        ix = 0
        for inputs, labels in dl_train:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.reshape(BATCH_SIZE, model.M, model.N, 3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss
            print(f"epoch progress: {(ix+1)/len(dl_train) * 100}%")
            ix += 1
        print(f"epoch {epoch}: loss {sum_loss}")
        torch.save(model, "models/weights")
       

    elapsed = time.time() - start
    print(f"Elapsed {elapsed // 60:.0f}m {elapsed % 60:.0f}s")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CoronaryNet()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, criterion, optimizer)
    torch.save(model, "models/weights")
