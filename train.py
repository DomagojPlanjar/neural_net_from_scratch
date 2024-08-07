import pickle
import time
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
import nn
import layers

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out'

# Network configuration, change as/if needed
config = {
    'max_epochs': 1,
    'batch_size': 50,
    'save_dir': SAVE_DIR,
    'weight_decay': 1e-3,
    'lr_policy': {
        1: {'lr': 1e-1},
        3: {'lr': 1e-2},
        5: {'lr': 1e-3},
        7: {'lr': 1e-4}
    }
}


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


def prepare_data():
    """
        Downloads the MNIST dataset and splits it into training, validation and test sets.
        Normalizes the data before returning (subtract the mean and divide by max).
    """
    ds_train = MNIST(DATA_DIR, train=True, download=True)
    ds_test = MNIST(DATA_DIR, train=False)

    train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
    train_y = ds_train.targets.numpy()
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]

    test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
    test_y = ds_test.targets.numpy()

    train_mean = train_x.mean()
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def build_network():
    """
    Builds the neural network architecture with L2 regularization using manually built layers from layers.py
    Feel free to modify this architecture as you wish, you can add or remove layers as needed.
    """
    net = []
    weight_decay = config['weight_decay']
    regularizers = []
    inputs = np.random.randn(config['batch_size'], 1, 28, 28)

    # Convolutional layers and operations
    net += [layers.Convolution(inputs, 16, 5, "conv1")]
    regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'conv1_l2reg')]
    net += [layers.MaxPooling(net[-1], "pool1")]
    net += [layers.ReLU(net[-1], "relu1")]
    net += [layers.Convolution(net[-1], 32, 5, "conv2")]
    regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'conv2_l2reg')]
    net += [layers.MaxPooling(net[-1], "pool2")]
    net += [layers.ReLU(net[-1], "relu2")]

    # Flatten layer
    net += [layers.Flatten(net[-1], "flatten3")]

    # Fully connected layers with L2 regularization
    net += [layers.FC(net[-1], 512, "fc3")]
    regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'fc3_l2reg')]
    net += [layers.ReLU(net[-1], "relu3")]
    net += [layers.FC(net[-1], 10, "logits")]

    data_loss = layers.SoftmaxCrossEntropyWithLogits()
    loss = layers.RegularizedLoss(data_loss, regularizers)

    return net, loss


def main():
    np.random.seed(int(time.time() * 1e6) % 2 ** 31)

    train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_data()
    save_dir = config['save_dir']

    # Load the trained model file if possible, else train the model and save it in save_dir
    try:
        with open(f"{save_dir}/trained_model.pkl", "rb") as f:
            net = pickle.load(f)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Training a new model.")
        net, regularized_loss = build_network()
        loss = regularized_loss
        net = nn.train(train_x, train_y, valid_x, valid_y, net, loss, config)
    else:
        # Use a dummy loss object since we are not training
        loss = layers.SoftmaxCrossEntropyWithLogits()

    # Evaluate the model on test data
    nn.evaluate("Test", test_x, test_y, net, loss, config)

    # Get one random image from test set, predict the label and show the results
    test_index = np.random.choice(len(test_x), 1, replace=False)[0]

    test_image = test_x[test_index:test_index+1]
    true_label = test_y[test_index]

    logits = nn.forward_pass(net, test_image)
    predicted_class = np.argmax(logits, axis=1)[0]

    plt.imshow(test_image[0,0,:,:], cmap='gray')
    plt.title(f"Predicted: {predicted_class}, True label: {np.argmax(true_label)}")
    plt.show()


if __name__ == '__main__':
    main()
