from tensorflow.keras.optimizers.schedules import ExponentialDecay

P = 4
K = 4
class_num = 5
epochs = 1000
hard = 3
emb_size = 128

batch_size = P * K

lr_schedule = ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps=10000,
    decay_rate=0.98)

model_path_old = "../../../../models/supervised_contrastive_loss_real0.8_v46/"
model_path = "../../../../models/supervised_contrastive_loss_real0.8_v46/"

