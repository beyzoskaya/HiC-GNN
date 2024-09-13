import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.loadtxt('Hi-C_dataset/CH12-LX/1mb/chr1_1mb.RAWobserved.txt')

bin_size = 1_000_000
data[:, 0] = data[:, 0] // bin_size  
data[:, 1] = data[:, 1] // bin_size  

max_bin = int(np.max(data[:, :2])) 
size = max_bin + 1  

contact_map = np.zeros((size, size))
for i, j, value in data:
    contact_map[int(i), int(j)] = value
    contact_map[int(j), int(i)] = value 

plt.figure(figsize=(10, 8))
sns.heatmap(contact_map, cmap="Blues", square=True)
plt.title('Hi-C Contact Map (1Mb Resolution)')
plt.xlabel('Genomic Bin (1Mb)')
plt.ylabel('Genomic Bin (1Mb)')
plt.show()
plt.savefig('contact_map_CH12-LX.png')
