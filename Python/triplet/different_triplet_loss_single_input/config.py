from tensorflow.keras.optimizers.schedules import ExponentialDecay

P = 4
K = 8
class_num = 5
epochs = 1000
hard = 3

batch_size = P * K

lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.95)

model_path = "../../../models/supervised_contrastive_loss/"
model_path_old = "../../../models/triplet_batch_hard/"