import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_2D_components(df, target_col, n_sample=100):
    cols = [c for c in df.columns if c != target_col]
    is_fraud = df[target_col]
    sample_1 = df[is_fraud].sample(n=n_sample, replace=False)
    sample_2 = df[~is_fraud].sample(n=n_sample, replace=False)
    sample_df = pd.concat([sample_1, sample_2])
    X = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(sample_df[cols])
    return X, sample_df[target_col]


def plot_scatter(X, y, palette_dict, title):
    plt.figure(figsize=(10, 10))
    for label in list(set(y)):
        X_l = X[y == label]
        plt.scatter(X_l[:, 0],
                    X_l[:, 1],
                    color=palette_dict[label],
                    s=1,
                    alpha=0.5,
                    label=label)
    plt.title(title)
    plt.legend()
    plt.show()