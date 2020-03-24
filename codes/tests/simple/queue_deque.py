import pandas as pd
import numpy as np

a = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]

df_rolling = pd.DataFrame(a).rolling(3).mean().fillna(0.0).values.squeeze(axis=1)

print(df_rolling)
