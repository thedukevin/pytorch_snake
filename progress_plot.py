
import numpy as np
import matplotlib.pyplot as plt

with open('progress.txt', 'r') as f:
    prog = [int(s[:-1]) for s in f.readlines()]

a = np.array(prog)
target_rate=50

plt.plot(a.cumsum())
plt.plot([0, len(a)], [0, len(a)*target_rate])
plt.savefig('progress')