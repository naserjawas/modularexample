import argparse
import os
import torch
import utils
import model.net as net 
from model.data_loader import test_data

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No Json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    model = net.NeuralNetwork().cuda() if params.cuda else net.NeuralNetwork()
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Angkle boot"
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        if params.cuda:
            x = x.cuda(non_blocking=True)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print("Predicted: {}, Actual: {}".format(predicted, actual))
