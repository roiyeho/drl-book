"""A set of plot functions used by other modules
    Author: Roi Yehoshua
    Date: June 2020
"""
import matplotlib.pyplot as plt

def plot_images(images, class_names, labels, file_name, n_cols=10):
    """Plot a sample of images from the training set
    :param images: the set of images
    :param class_names: mapping from class indices to names
    :param labels: the labels of the images
    :param file_name: the file where the figure will be saved
    :param n_cols: number of the columns in the plot
    :return:
    """
    n_rows = len(images) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    k = 0
    for ax in axes.flat:
        ax.imshow(images[k])
        ax.axis('off')
        ax.set_title(class_names[labels[k][0]], fontsize=16)  # each label is a 1D vector of size 1
        k += 1
    plt.savefig(f'figures/{file_name}.png')
    plt.close()

def plot_learning_curve(training_acc, validation_acc, file_name):
    """Plot the learning curve with the accuracy on the training and validation sets
    :param training_acc: the accuracy on the training set
    :param validation_acc: the accuracy on the validation set
    :file_name the name of the file where the figure will be saved
    :return:
    """
    x = range(1, len(training_acc) + 1)
    plt.plot(x, training_acc, label='Training')
    plt.plot(x, validation_acc, label='Validation')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f'figures/{file_name}.png')
    plt.close()

def plot_predictions(images, class_names, true_labels, predicted_labels, file_name):
    """ Plot the given images from the test set, their true labels and predicted labels
    :param images: the set of images
    :param class_names: mapping from class indices to names
    :param true_labels: the true labels of the images
    :param predicted_labels: the predicted labels by the model
    :param file_name: name of the file where the figure will be saved
    :return:
    """
    fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
    k = 0
    for ax in axes.flat:
        ax.imshow(images[k])
        ax.axis('off')
        title = f'{class_names[true_labels[k][0]]} ({class_names[predicted_labels[k]]})'
        ax.set_title(title, fontsize=12)
        k += 1
    plt.savefig(f'figures/{file_name}.png')
    plt.close()