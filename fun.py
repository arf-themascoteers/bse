import pandas as pd

df = pd.read_csv("lucas.csv")
df = df.sample(frac=0.01).reset_index(drop=True)
df.to_csv("lucas_min.csv", index=False)