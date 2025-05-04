import argparse
import os
import utils
import torch
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y), in enumerate(dataloader):
        if params.cuda:
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print("loss: {:7f} [{:5d}/{:5d}]".format(loss, current, size))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            if params.cuda:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print("Test Error: \n Accuracy: {:0.1f}, Avg loss: {:8f}".format(100*correct, test_loss))

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, loss_fn, params):
    for t in range(params.num_epochs):
        print("Epoch {}\n............................".format(t+1))
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch model state to model.pth")

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No Json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()
    print("Cuda available: {}".format(params.cuda))

    model = net.NeuralNetwork().cuda() if params.cuda else net.NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    dl = data_loader.fetch_dataloader(['train', 'test'], params)
    train_dataloader = dl['train']
    test_dataloader = dl['test']

    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, loss_fn, params)
