import numpy as np
import torch
from datasets import ThingsMEGDataset

X = ThingsMEGDataset(data_dir='./data', split='train').X
mean = X.mean(dim=(0, 2))
std = X.std(dim=(0, 2))
np.savetxt('./src/mean.csv', mean)
np.savetxt('./src/std.csv', std)