import os
import math
import pickle

import numpy as np
import skimage as ski
import skimage.io
from skimage import img_as_ubyte


def forward_pass(net, inputs):
    """
    Perform a forward pass through the network.

    Parameters:
    net (list): List of layers in the network.
    inputs (np.ndarray): Input data.

    Returns:
    np.ndarray: Output after passing through the network.
    """
    output = inputs
    for layer in net:
        output = layer.forward(output)
    return output


def backward_pass(net, loss, x, y):
    """
    Perform a backward pass through the network.

    Parameters:
    net (list): List of layers in the network.
    loss (Loss): Loss function object.
    x (np.ndarray): Input data.
    y (np.ndarray): Ground truth labels.

    Returns:
    list: List of gradients for each layer.
    """
    grads = []
    grad_out = loss.backward_inputs(x, y)

    if loss.has_params:
        grads += loss.backward_params()

    for layer in reversed(net):
        grad_inputs = layer.backward_inputs(grad_out)
        if layer.has_params:
            grads += [layer.backward_params(grad_out)]
        grad_out = grad_inputs

    return grads


def sgd_update_params(grads, config):
    """
    Update the network parameters using stochastic gradient descent.

    Parameters:
    grads (list): List of gradients for each layer.
    config (dict): Configuration dictionary containing learning rate.
    """
    lr = config['lr']

    for layer_grads in grads:
        for i in range(len(layer_grads) - 1):
            params = layer_grads[i][0]
            grads = layer_grads[i][1]
            params -= lr * grads


def draw_conv_filters(epoch, step, layer, save_dir):
    """
    Draw and save convolution filters.

    Parameters:
    epoch (int): Current epoch number.
    step (int): Current step number.
    layer (Layer): Convolutional layer.
    save_dir (str): Directory to save the filter images.
    """
    C = layer.C
    w = layer.weights.copy()
    num_filters = w.shape[0]
    k = int(np.sqrt(w.shape[1] / C))
    w = w.reshape(num_filters, C, k, k)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border

    for i in range(1):
        img = np.zeros([height, width])

        for j in range(num_filters):
            r = int(j / cols) * (k + border)
            c = int(j % cols) * (k + border)
            img[r:r + k, c:c + k] = w[j, i]

        filename = f'{layer.name}_epoch_{epoch:02d}_step_{step:06d}_input_{i:03d}.png'
        ski.io.imsave(os.path.join(save_dir, filename), img_as_ubyte(img))


def train(train_x, train_y, valid_x, valid_y, net, loss, config):
    """
    Train the network.

    Parameters:
    train_x (np.ndarray): Training data inputs.
    train_y (np.ndarray): Training data labels.
    valid_x (np.ndarray): Validation data inputs.
    valid_y (np.ndarray): Validation data labels.
    net (list): List of layers in the network.
    loss (Loss): Loss function object.
    config (dict): Configuration dictionary.

    Returns:
    list: Trained network.
    """
    lr_policy = config['lr_policy']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    num_examples = train_x.shape[0]

    assert num_examples % batch_size == 0

    num_batches = num_examples // batch_size

    for epoch in range(1, max_epochs + 1):
        solver_config = lr_policy.get(epoch, lr_policy[1])
        cnt_correct = 0

        permutation_idx = np.random.permutation(num_examples)
        train_x = train_x[permutation_idx]
        train_y = train_y[permutation_idx]
        epoch_loss = []

        for i in range(num_batches):
            batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
            batch_y = train_y[i * batch_size:(i + 1) * batch_size, :]

            logits = forward_pass(net, batch_x)
            loss_val = loss.forward(logits, batch_y)
            epoch_loss.append(loss_val)

            yp = np.argmax(logits, 1)
            yt = np.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()
            grads = backward_pass(net, loss, logits, batch_y)
            sgd_update_params(grads, solver_config)

            if i % 5 == 0:
                print(f'epoch = {epoch}, step = {i * batch_size}/{num_examples}, batch loss = {np.mean(epoch_loss)}')

            if i % 100 == 0:
                draw_conv_filters(epoch, i * batch_size, net[0], save_dir)
                draw_conv_filters(epoch, i * batch_size, net[3], save_dir)

            if i > 0 and i % 50 == 0:
                print(f"Train accuracy = {cnt_correct / ((i + 1) * batch_size) * 100:.2f}")

        print(f"Train accuracy = {cnt_correct / num_examples * 100:.2f}")
        evaluate("Validation", valid_x, valid_y, net, loss, config)

    with open(f"{save_dir}/trained_model.pkl", "wb") as f:
        pickle.dump(net, f)

    return net


def evaluate(name, x, y, net, loss, config):
    """
    Evaluate the network.

    Parameters:
    name (str): Name of the evaluation phase (e.g., "Validation").
    x (np.ndarray): Input data.
    y (np.ndarray): Ground truth labels.
    net (list): List of layers in the network.
    loss (Loss): Loss function object.
    config (dict): Configuration dictionary.
    """
    print("\nRunning evaluation:", name)

    batch_size = config['batch_size']
    num_examples = x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    loss_avg = 0

    for i in range(num_batches):
        batch_x = x[i * batch_size:(i + 1) * batch_size, :]
        batch_y = y[i * batch_size:(i + 1) * batch_size, :]

        logits = forward_pass(net, batch_x)
        yp = np.argmax(logits, 1)
        yt = np.argmax(batch_y, 1)
        cnt_correct += (yp == yt).sum()
        loss_val = loss.forward(logits, batch_y)
        loss_avg += loss_val

    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches
    print(f"{name} accuracy = {valid_acc:.2f}")
    print(f"{name} avg loss = {loss_avg:.2f}\n")
