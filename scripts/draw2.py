import matplotlib.pyplot as plt
import numpy as np
  
y1 = [104, 28, 37, 55, 41, 95, 28]
y2 = [154, 52, 43, 74, 64, 110, 29]
labels = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

plt.title('Number of flipped wins (per topic)')

x = np.arange(len(y1))

width = 0.40
  
plt.bar(x-0.2, y1, width, label="baseline")
plt.bar(x+0.2, y2, width, label="our method")

plt.xticks(x, labels, rotation=40)

plt.tight_layout()
plt.legend(loc="upper right")
plt.savefig('draw2.png')
