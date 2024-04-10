import pandas as pd
from matplotlib import pyplot as plt
import argparse

#parser = argparse.ArgumentParser("dir")
dataDir = "6lu7/log336h_6lu7/data10/"
cols = ["Docking_Score", "QED_score", "Toxicity_Probability"]
df = pd.read_csv(dataDir+"present/scores.csv", names=cols)

for col in cols:
    fig, ax = plt.subplots(ncols=1,nrows=1)
    fig = df.hist(column=col, ax=ax)
    ax.set_ylabel("number of candidates")
    ax.set_xlabel(col)
    plt.savefig(dataDir+f"present/{col}.png")
