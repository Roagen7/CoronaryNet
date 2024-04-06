import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import load_train, load_test
from model import CoronaryNet, CoronaryNet2
from loss import arc_loss, manifold_likelihood_loss


LEARNING_RATE=5e-4
EPOCHS=300
BATCH_SIZE=1


def train_model(model, criterion, optimizer, device, 
dl_train=load_train(BATCH_SIZE)[1], 
dl_test=load_test(BATCH_SIZE)[1],
save_path="./models/weights", 
verbose=True, 
batch_size=BATCH_SIZE,
epochs=EPOCHS):
    start = time.time()
    min_val_loss = torch.inf
    for epoch in range(epochs):
        model.train()
        sum_loss = 0
        ix = 0
        for inputs, params, labels in dl_train:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            params = params.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs, params)
            outputs = outputs.reshape(batch_size, model.M, model.N, 3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss
            if verbose: print(f"epoch progress: {(ix+1)/len(dl_train) * 100}% loss: {loss}")
            ix += 1

        model.eval()
        ix = 0
        val_loss = 0

        with torch.no_grad():
            for inputs, params, labels in dl_test:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                params = params.to(device, dtype=torch.float)
                outputs = model(inputs, params)
                outputs = outputs.reshape(batch_size, model.M, model.N, 3)
                loss = criterion(outputs, labels)
                val_loss += loss
                if verbose: print(f"val progress: {(ix+1)/len(dl_test) * 100}% loss: {loss}")
                ix += 1

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print(f"best loss! saved as {save_path}_best")
            torch.save(model, f"{save_path}_best")
        torch.save(model, save_path)

        print(f"epoch {epoch+1}: loss {sum_loss} val_loss {val_loss}")
        
       

    elapsed = time.time() - start
    print(f"Elapsed {elapsed // 60:.0f}m {elapsed % 60:.0f}s")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CoronaryNet2(instance_batch=BATCH_SIZE==1)
    model.to(device)

    criterion = arc_loss(lam=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)

    _, dl_train = load_train(BATCH_SIZE, size=100)
    _, dl_test = load_test(BATCH_SIZE, size=10)

    train_model(model, criterion, optimizer, dl_train=dl_train, device=device)
