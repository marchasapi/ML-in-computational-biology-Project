import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plotHistogramForFeature(df, feature, ytitle, bins = 'auto'):
    plt.figure(figsize=(10, 6))

    if bins == None:
        plt.hist(df[feature], color='skyblue', edgecolor='black')
    else:
        plt.hist(df[feature], bins=bins, color='skyblue', edgecolor='black')

    plt.title('Distribution of ' + feature)
    plt.xlabel(feature)
    plt.ylabel(ytitle)
    plt.grid(True)
    plt.show()
