import sys
import pandas as pd
from bm_plot_utils import main

table = pd.read_csv(sys.argv[1], sep="\t")

main(table)
