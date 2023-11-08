import numpy as np
import random
# from memory_profiler import profile
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import sys


def tripletDataset_balanced_synAnchor(data_real, data_syn, labels):
    """
    construct triplets: anchor (synthetic), positive (real data in the same class), negative (real data in different classes).
    balanced number of positives and negatives.
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


def tripletDataset_balanced_realAnchor(data_real, labels_real, data_syn, labels_syn):
    """
    construct triplets: anchor (synthetic), positive (real data in the same class), negative (real data in different classes)
    balanced number of positives and negatives.
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
    for i in range(len(data_real)):
        x_anchor = data_real[i, :, :]

        y = labels_real[i]
        indices_for_pos = np.squeeze(np.where(labels_syn == y))
        indices_for_neg = np.squeeze(np.where(labels_syn != y))
        for j in range(len(indices_for_pos)):
            x_positive = data_syn[indices_for_pos[j], :, :]
            x_negative = data_syn[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)], :, :]
            # for k in range(len(indices_for_neg)):
            #     x_negative = data_syn[indices_for_neg[k], :, :]
            anchors.append(x_anchor)
            positives.append(x_positive)
            negatives.append(x_negative)
    print(f"number of triplets: {len(anchors)}")
    return [np.array(anchors), np.array(positives), np.array(negatives)]


def tripletDataset_realAnchor_2(data_real, labels_real, data_syn, labels_syn):
    """
    construct triplets: anchor (synthetic), positive (real data in the same class), negative (real data in different classes)
    for each anchor, positive pairs, randomly select one negative from each other class.
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
    for i in range(len(data_real)):
        x_anchor = data_real[i, :, :]

        y = labels_real[i]
        indices_for_pos = np.squeeze(np.where(labels_syn == y))
        indices_for_neg = np.squeeze(np.where(labels_syn != y))
        for j in range(len(indices_for_pos)):
            x_positive = data_syn[indices_for_pos[j], :, :]
            labels_neg = labels_syn[indices_for_neg]
            labels_set = set(labels_neg)
            for l in labels_set:
                ind_l = np.squeeze(np.where(labels_syn == l))
                x_negative = data_syn[ind_l[random.randint(0, len(ind_l) - 1)], :, :]
             
                anchors.append(x_anchor)
                positives.append(x_positive)
                negatives.append(x_negative)
    print(f"number of triplets: {len(anchors)}")
    return [np.array(anchors), np.array(positives), np.array(negatives)]


def tripletDataset_all_realAnchor(data_real, labels_real, data_syn, labels_syn):
    """
    construct triplets: anchor (synthetic), positive (real data in the same class), negative (real data in different classes)
    balanced number of positives and negatives.
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
    for i in range(len(data_real)):
        x_anchor = data_real[i, :, :]

        y = labels_real[i]
        indices_for_pos = np.squeeze(np.where(labels_syn == y))
        indices_for_neg = np.squeeze(np.where(labels_syn != y))
        for j in range(len(indices_for_pos)):
            x_positive = data_syn[indices_for_pos[j], :, :]
            # x_negative = data_syn[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)], :, :]
            for k in range(len(indices_for_neg)):
                x_negative = data_syn[indices_for_neg[k], :, :]
                anchors.append(x_anchor)
                positives.append(x_positive)
                negatives.append(x_negative)
    print(f"number of triplets: {len(anchors)}")
    return [np.array(anchors), np.array(positives), np.array(negatives)]


def create_batch(data_real, labels_real, data_syn, labels_syn, batch_size):
    """
    construct triplets: anchor (synthetic), positive (real data in the same class), negative (real data in different classes)
    balanced number of positives and negatives.
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
    for i in range(batch_size):
        random_index = random.randint(0, len(data_real)-1)
        x_anchor = data_real[random_index, :, :]
        y = labels_real[random_index]
        
        indices_for_pos = np.squeeze(np.where(labels_syn == y))
        indices_for_neg = np.squeeze(np.where(labels_syn != y))
        
        x_positive = data_syn[indices_for_neg[random.randint(0, len(indices_for_pos) - 1)], :, :]
        x_negative = data_syn[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)], :, :]
        
        anchors.append(x_anchor)
        positives.append(x_positive)
        negatives.append(x_negative)
        
    return [np.array(anchors), np.array(positives), np.array(negatives)]

def tripletDataset_allTriplets(filename, sample_num, data_real, data_syn, labels):
    """
    construct triplets: anchor (real), positive (synthetic data in the same class), negative (synthetic data in different classes). ( , 28*3, 52)
    input real and synthetic data are in the same order (aligned).
    traverse all data to get as more triplets as we can.
    Args:
        filename: file name to save triplet data
        sample_num: number of triplets
        data_real:
        data_syn:
        labels:

    Returns:
    """
    
    # use np.memmap to prevent CPU memory explosion
    triplets = np.memmap(
        filename,
        dtype='float32',
        mode='w+',
        shape=(sample_num, 28*3, 52)
    )
    
    cnt = 0
    for i in range(len(data_real)):
        x_anchor = data_real[i, :, :]

        y = labels[i]
        indices_for_pos = np.squeeze(np.where(labels == y))
        indices_for_neg = np.squeeze(np.where(labels != y))
        for j in range(len(indices_for_pos)):
            x_positive = data_syn[indices_for_pos[j], :, :]
            for k in range(len(indices_for_neg)):
                x_negative = data_syn[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)], :, :]
                triplet = np.vstack((np.vstack((x_anchor, x_positive)), x_negative))
                triplets[cnt,] = triplet
                cnt += 1
    triplets.flush()


def data_dump(data_path="../../models/triplet/"):
    X_real = np.load(os.path.join(data_path, "X.npy"))
    y_real = np.load(os.path.join(data_path, "Y.npy")) - 1
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_train_r, y_train_r, test_size=0.1, random_state=42)
   
    X_syn = np.load(os.path.join(data_path, "X_syn.npy"))
    y_syn = np.load(os.path.join(data_path, "Y_syn.npy")) - 1
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_s, y_train_s, test_size=0.1, random_state=42)
    
    X_train = tripletDataset_balanced_realAnchor(X_train_r, y_train_r, X_train_s, y_train_s)
    X_test = tripletDataset_balanced_realAnchor(X_test_r, y_test_r, X_test_s, y_test_s)
    X_val = tripletDataset_balanced_realAnchor(X_val_r, y_val_r, X_val_s, y_val_s)
    """
        triplet shape:
        train: 4768320
        test: 111688
        validation: 7186
    """
    tripletDataset_allTriplets(os.path.join("../../models/triplet_v2/", "train_triplets.dat"), 4768320, X_train_r,
                               X_train_s, y_train_r)
    tripletDataset_allTriplets(os.path.join("../../models/triplet_v2/", "test_triplets.dat"), 111688, X_test_r,
                               X_test_s, y_test_r)
    tripletDataset_allTriplets(os.path.join("../../models/triplet_v2/", "val_triplets.dat"), 7186, X_val_r, X_val_s,
                               y_val_r)
    
def cal_dis(a, p):
	return tf.reduce_mean(tf.square(a - p))
 

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, triplet_dict, emb_size, net=None, alpha=0.2, batch_size=32, dim=(28, 52), mode="all",
                 shuffle=True, data_real=None, labels_real=None, data_syn=None, labels_syn=None):
        """

        Args:
            list_IDs:
            triplet_dict: dictionary, "filename", "shape" of triplet dataset
            batch_size:
            dim: dimension of anchor/positive/negative
            shuffle:
        """
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = np.arange(triplet_dict["shape"][0])
        self.shuffle = shuffle
        self.triplets = np.memmap(
            triplet_dict["filename"],
            dtype="float32",
            mode='r',
            shape=triplet_dict["shape"]
        )
        self.emb_size = emb_size
        self.on_epoch_end()
        self.net = net
        self.emb_size = 128
        self.alpha = alpha
        self.data_real = data_real
        self.data_syn = data_syn
        self.labels_real = labels_real
        self.labels_syn = labels_syn
        self.mode = mode
        self.semi_cnt = 0
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        if self.mode == "all":
            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            
            # Generate data
            X, y = self.__data_generation(list_IDs_temp)
        
        if self.mode == "semi":
            # indexes = np.zeros(self.batch_size)
            list_IDs_temp = []
            d = self.dim[0]
            for i in range(len(indexes)):
                id = self.list_IDs[indexes[i]]
                triplet = self.triplets[id]
                anchor = triplet[:d]
                positive = triplet[d:2 * d]
                negative = triplet[2 * d:3 * d]
                emb = self.net.predict([anchor[np.newaxis,], positive[np.newaxis,], negative[np.newaxis,]])[0]
                emb_a = emb[:self.emb_size]
                emb_p = emb[self.emb_size:self.emb_size * 2]
                emb_n = emb[self.emb_size * 2:self.emb_size * 3]
                dis_ap = cal_dis(emb_a, emb_p)
                dis_an = cal_dis(emb_a, emb_n)
                
                if dis_ap < dis_an and dis_an < (dis_ap + self.alpha / 2):
                    list_IDs_temp.append(id)
            
            if len(list_IDs_temp) == 0:
                X = create_batch(self.data_real, self.labels_real, self.data_syn, self.labels_syn,
                                 self.batch_size - len(list_IDs_temp))
                y = np.zeros((self.batch_size, 3 * self.emb_size))
            else:
                # Generate data
                X, _ = self.__data_generation(list_IDs_temp)
                if len(list_IDs_temp) < self.batch_size:
                    X_random = create_batch(self.data_real, self.labels_real, self.data_syn, self.labels_syn,
                                            self.batch_size - len(list_IDs_temp))
                    for k in range(3):
                        X[k] = np.vstack((X[k], X_random[k]))
                y = np.zeros((self.batch_size, 3 * self.emb_size))
            self.semi_cnt += len(list_IDs_temp)
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        try:
            print(f"semi-hard triplets: {self.semi_cnt}")
        except Exception as e:
            pass
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [np.empty((self.batch_size, *self.dim)), np.empty((self.batch_size, *self.dim)),
             np.empty((self.batch_size, *self.dim))]
        y = np.zeros((self.batch_size, 3 * self.emb_size))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            for j in range(3):
                X[j][i,] = self.triplets[i, j * self.dim[0]:(j + 1) * self.dim[0], :]
            # X[i,] = np.load('data/' + ID + '.npy')
        
        return X, y


if __name__ == "__main__":
    data_dump()
    triplets = np.memmap(
        "../../models/triplet_v2/train_triplets.dat",
        dtype='float32',
        mode='r',
        shape=(4768320, 28 * 3, 52)
    )
    print(1)