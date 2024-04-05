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


def train_model(model, criterion, optimizer, device, 
dl_train=load_train(BATCH_SIZE)[1], 
save_path="./models/weights", 
verbose=True, 
batch_size=BATCH_SIZE,
epochs=EPOCHS):
    start = time.time()
    model.train()
    for epoch in range(epochs):
        sum_loss = 0
        ix = 0
        for inputs, labels in dl_train:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.reshape(batch_size, model.M, model.N, 3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss
            if verbose: print(f"epoch progress: {(ix+1)/len(dl_train) * 100}% loss: {loss}")
            ix += 1
        print(f"epoch {epoch+1}: loss {sum_loss}")
        torch.save(model, save_path)
       

    elapsed = time.time() - start
    print(f"Elapsed {elapsed // 60:.0f}m {elapsed % 60:.0f}s")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CoronaryNet()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    _, dl_train = load_train(BATCH_SIZE, size=100)

    train_model(model, criterion, optimizer, dl_train=dl_train, device=device)
    torch.save(model, "./models/weights")
