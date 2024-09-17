
import pandas as pd
from glob import glob
import sys 

files_csv = pd.concat([pd.read_csv(el, index_col=0) for el in glob("*.csv")], axis=1)
files_csv.sort_index()


scores_cols = []
for el in glob("*.txt"):
    name = el.split(".")[0]
    scores = pd.read_csv(el, header=None)
    scores.columns = [name]
    scores_cols.append(scores)

scores_cols = pd.concat(scores_cols, axis=1)
scores_cols.to_csv(f"{sys.argv[1]}_scores.csv")
files_csv.to_csv(f"{sys.argv[1]}.csv")
