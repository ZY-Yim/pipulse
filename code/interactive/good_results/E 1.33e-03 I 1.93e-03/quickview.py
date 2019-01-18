import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

space = np.load("space.npy")
metric = np.load("metric.npy")

#sns.heatmap(space[:,-100:])
#plt.show()

df = pd.DataFrame(space[:,-100:])
model = KMeans(n_clusters=9)
pred = model.fit_predict(df)
order = np.argsort(pred)
fig = plt.figure(figsize=(15,10))
g = sns.heatmap(space[order,-100:],yticklabels=2,xticklabels=100,cmap='viridis')
g.set_yticklabels(fig.axes[0].get_yticklabels(),rotation=0)

#sns.heatmap(metric)
plt.show()
