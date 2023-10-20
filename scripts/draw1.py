import matplotlib.pyplot as plt
import numpy as np
  
y1 = [55, 12, 11, 29, 12, 47, 12]
y2 = [89, 17, 20, 37, 35, 64, 8]
labels = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

plt.title('Number of times agent flags an evidence as relevant (per topic)')

x = np.arange(len(y1))

width = 0.40
  
plt.bar(x-0.2, y1, width, label="underperformed baseline")
plt.bar(x+0.2, y2, width, label="outperformed baseline")

plt.xticks(x, labels, rotation=40)

plt.tight_layout()
plt.legend(loc="upper right")
plt.savefig('draw1.png')
