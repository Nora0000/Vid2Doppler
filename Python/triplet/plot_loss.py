import os.path

from matplotlib import pyplot as plt
import numpy as np

model_path = "../../models/triplet_v56/"

losses = []
for bs in [32, 16]:
	loss = np.load(os.path.join(model_path, f"loss_{bs}.npy"))
	if len(losses) == 0:
		losses = loss
	else:
		losses = np.hstack((losses, loss))
plt.plot(losses[0, :])
plt.plot(losses[1, :])
plt.legend(['Training', 'Validation'])
plt.title("loss during training")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()