import numpy as np
import random
from memory-profiler

def tripletDataset(data_real, data_syn, labels):
    """
    construct triplets: anchor (synthetic), positive (real data in the same class), negative (real data in different classes)
    input real and synthetic data are in the same order (aligned).
    Args:
        data_real:
        data_syn:
        labels:

    Returns:

    """
    anchors = []
    positives = []
    negatives = []
    for i in range(len(data_syn)):
        x_anchor = data_syn[i, :, :]

        y = labels[i]
        indices_for_pos = np.squeeze(np.where(labels == y))
        indices_for_neg = np.squeeze(np.where(labels != y))
        for j in range(len(indices_for_pos)):
            x_positive = data_real[indices_for_pos[j], :, :]
            x_negative = data_real[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)], :, :]
            anchors.append(x_anchor)
            positives.append(x_positive)
            negatives.append(x_negative)
    return [np.array(anchors), np.array(positives), np.array(negatives)]


def tripletDataset_v2(data_real, data_syn, labels):
    """
    construct triplets: anchor (real), positive (synthetic data in the same class), negative (synthetic data in different classes)
    input real and synthetic data are in the same order (aligned).
    traverse all data to get as more triplets as we can.
    Args:
        data_real:
        data_syn:
        labels:

    Returns:
    """
    anchors = []
    positives = []
    negatives = []
    for i in range(len(data_real)):
        x_anchor = data_real[i, :, :]

        y = labels[i]
        indices_for_pos = np.squeeze(np.where(labels == y))
        indices_for_neg = np.squeeze(np.where(labels != y))
        for j in range(len(indices_for_pos)):
            x_positive = data_syn[indices_for_pos[j], :, :]
            for k in range(len(indices_for_neg)//2):
                x_negative = data_syn[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)], :, :]
                # x_negative = data_syn[indices_for_neg[k], :, :]
                anchors.append(x_anchor)
                positives.append(x_positive)
                negatives.append(x_negative)
    print("number of triplets: {}".format(len(anchors)))
    return [np.array(anchors), np.array(positives), np.array(negatives)]


def data_generator(data, batch_size, emb_size):
    """
    yield a batch of data to save GPU memory
    Args:
        data: triplet dataset
        batch_size:
    """
    anchors = data[0]
    positives = data[1]
    negatives = data[2]
    batch_y = np.zeros((batch_size, 3*emb_size))
    num_samples = len(data[0])
    while True:
    # Shuffle the data at the start of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        anchors = anchors[indices]
        positives = positives[indices]
        negatives = negatives[indices]

        for i in range(0, num_samples, batch_size):
            anchor_batch = anchors[i:i+batch_size, :, :]
            positive_batch = positives[i:i+batch_size, :, :]
            negative_batch = negatives[i:i+batch_size, :, :]
            batch_data = [anchor_batch, positive_batch, negative_batch]
            yield batch_data, batch_y




