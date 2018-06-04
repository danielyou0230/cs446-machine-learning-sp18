"""Generative Adversarial Networks
"""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan
import itertools


def train(model, mnist_dataset, learning_rate=0.0005, batch_size=16,
          num_steps=5000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1, 1, [batch_size, 2])
        # Train generator and discriminator
        _, d_loss = model.session.run([model.d_op, model.d_loss],
                                      feed_dict={
                                      model.lr_placeholder: learning_rate,
                                      model.x_placeholder: batch_x,
                                      model.z_placeholder: batch_z
                                      })

        _, g_loss = model.session.run([model.g_op, model.g_loss],
                                      feed_dict={
                                      model.lr_placeholder: learning_rate,
                                      model.z_placeholder: batch_z
                                      })
        print("Step: {:5d}/{:5d} | Discriminator Loss: {:.5f} | Generator Loss: {:.5f}"
              .format(step + 1, num_steps, d_loss, g_loss))
        if step % 500 == 0:
            plot_result(model, step)


def plot_result(model, step):
    res = 10
    print("Plotting at step {0}".format(step))
    # Plot
    # coordinate_x = np.linspace(-1, 1, 20)
    # coordinate_y = np.linspace(-1, 1, 20)
    coordinates = [itr for itr in range(res + 1)]
    # grid_x, grid_y = np.meshgrid(coordinate_x, coordinate_y)
    # grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    grid = (np.array([[x, y] for x in coordinates for y in coordinates]) - res / 2.) / (res / 2.)
    # Predict
    img = model.session.run(model.x_hat, feed_dict={model.z_placeholder: grid})
    # img = model.session.run(model.x_hat, feed_dict={model.z_placeholder: batch_z})
    # out = np.array([itr.reshape((28, 28)) for itr in img])
    out = np.empty((28 * (res + 1), 28 * (res + 1)))
    for x in coordinates:
        for y in coordinates:
            # print(x * 21 + y,)
            out[x * 28:(x + 1) * 28,
                y * 28:(y + 1) * 28] = img[x * (res + 1) + y,:].reshape(28, 28)
    # print(grid)
    # print(out.shape)
    plt.imsave('latent_space_gan_{0}.png'.format(step), out, cmap="gray")


def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan()

    # Start training
    train(model, mnist_dataset)

    #
    plot_result(model, "final")


if __name__ == "__main__":
    tf.app.run()
