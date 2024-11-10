import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


df = pd.read_csv(r'results.csv')
df["train_size"] = (df["train_size"] * 100).astype(int)
skip_df = df[df["manual"]==0].copy()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
axes = axes.flatten()

algorithms = skip_df["algorithm"].unique()
algorithms = ["lr"]
algorithm_label = {
    "lr":"Linear Regression",
    "mlp":"ANN",
    "rf":"Random Forest",
    "svr":"Support Vector Machine"
}
colors = cm.jet(np.linspace(0, 1, 11))

for index, a in enumerate(algorithms):
    a_df = skip_df[skip_df["algorithm"] == a].copy()
    ax = axes[index]
    ax.set_xlabel('Number of bands', fontsize=18)
    ax.set_ylabel(r"$R^2$", fontsize=18)
    #ax.set_xticks([i for i in range(1,100,10)])
    ax.tick_params(axis='both', labelsize=14)

    train_size = sorted(a_df["train_size"].unique())

    for c, bands in enumerate(train_size):
        that_df = a_df[a_df["train_size"] == bands].copy()
        ax.plot(that_df['n_bands'], that_df["R^2"],
                label=str(bands),
                color=colors[c],
                fillstyle='none', markersize=7, linewidth=2)

    if index == 0:
        legend = ax.legend(loc='upper left', ncols=15,bbox_to_anchor=(0,1.2),title='Train size (%)')
        legend.get_title().set_fontsize('12')
        legend.get_title().set_fontweight('bold')
    ax.set_title(algorithm_label[a], fontsize=20)

plt.savefig("plot2.png", bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.close(fig)

