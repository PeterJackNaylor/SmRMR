
import pandas as pd
from glob import glob
import sys 

files_csv = pd.concat([pd.read_csv(el, index_col=0) for el in glob("*.csv")], axis=1)
files_csv.to_csv(f"{sys.argv[1]}.csv")
