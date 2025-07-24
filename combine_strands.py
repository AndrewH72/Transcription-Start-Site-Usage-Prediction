# %%
import os
import re
import pandas as pd
import numpy as np
import polars as pl
from natsort import natsorted

# %%
# Dictionary to hold strand pairs.
strand_pairs = {
    # GM12878
    "ENCFF074SXQ": "ENCFF164VLA", # from experiment ENCSR000AED; file format conversion: signal generation (alignment)
    "ENCFF546NVF": "ENCFF182LTN", # from experiment ENCSR00AED; file format conversion: file format conversion (signal generation and chromosome sizes)
    "ENCFF078ATR": "ENCFF037DUE", # from experiment ENCSR000AEF; file format conversion: signal generation (alignment)
    "ENCFF892WMR": "ENCFF985TNZ", # from experiment ENCSR00AEF; file format conversion: file format conversion (signal generation and chromosome sizes)

    # K562
    "ENCFF829PNJ": "ENCFF336COA", # from experiment ENCSR000AEM; file format conversion: signal generation (alignment)
    "ENCFF964BAP": "ENCFF006DQI", # from experiment ENCSR000AEM; signal generation: signal generation (signal generation and chromosome sizes) 
    "ENCFF777EAJ": "ENCFF040DXX", # from experiment ENCSR000AEO; signal generation: file format conversion (alignment)
    "ENCFF528VFJ": "ENCFF097ASF", # from experiment ENCSR000AEO; signal generation: file format conversion (signal generation and  chromosome sizes)
}

# %%
# Splitting .tab and .txt files into different arrays.
# Sorting each array first by whether it is a plus or minus and then by its cell line.
data = os.listdir("/home/coder/data-REU")
tab_files = sorted([file for file in data if file.endswith(".tab") and "unstranded" not in file], key=lambda x: (x[x.find(".") + 1], x[0]))
coverage_files = sorted([file for file in data if file.endswith(".txt") and "unstranded" not in file], key=lambda x: (x[x.find(".") + 1], x[0]))

# %%
# Loop to combine the signals of the minus and plus strands into a single dataset for .tab files.
for i in range(len(tab_files) // 2):
   # Identify the cell line splicing the string up to the first "."
   # Identify the minus strand's and plus strand's name, the ENCFF*.
   cell_line = tab_files[i][ :tab_files[i].find(".")]
   key_file_name = re.search(r"E[a-zA-Z0-9]+", tab_files[i]).group(0)
   for j in range(len(tab_files) // 2, len(tab_files)):
       value_file_name = re.search(r"E[a-zA-Z0-9]+", tab_files[j]).group(0)

       # Create dataframes for our plus and minus strand data.
       # Rename the columns, so the data will actually add.
       if strand_pairs[key_file_name] == value_file_name:
           # key_df = pd.read_csv(f"/home/coder/data-REU/{tab_files[i]}", sep="\t", skiprows=2).fillna(0)
           # value_df = pd.read_csv(f"/home/coder/data-REU/{tab_files[j]}", sep="\t", skiprows=2).fillna(0)
           key_df = pd.DataFrame(pl.read_csv(f"/home/coder/data-REU/{tab_files[i]}", separator="\t", skip_rows=2)).fillna(0)
           value_df = pd.DataFrame(pl.read_csv(f"/home/coder/data-REU/{tab_files[j]}", separator="\t", skip_rows=2)).fillna(0)

           res_df = key_df.add(value_df)
           res_df.to_csv(f"/home/coder/REU/stranded_files/{cell_line}.stranded.{value_file_name}.{key_file_name}_values_TSS.tab", sep="\t", index=False)

# %%
# Loop to combine the signals of the minus and plus strands into a single dataset for .txt files.
for i in range(len(coverage_files) // 2):
    # Identify the cell line splicing the string up to the first "."
    # Identify the minus strand's and plus strand's name, the ENCFF*.
    cell_line = coverage_files[i][ :coverage_files[i].find(".")]
    key_file_name = re.search(r"E[a-zA-Z0-9]+", coverage_files[i]).group(0)
    for j in range(len(coverage_files) // 2, len(coverage_files)):
        value_file_name = re.search(r"E[a-zA-Z0-9]+", coverage_files[j]).group(0)
        # Create dataframes for our plus and minus strand data.
        # Rename the columns, so the data will actually add.
        if strand_pairs[key_file_name] == value_file_name:
            # key_df = pd.read_csv(f"/home/coder/data-REU/{coverage_files[i]}", sep="\t")
            # value_df = pd.read_csv(f"/home/coder/data-REU/{coverage_files[j]}", sep="\t")
            key_df = pd.DataFrame(pl.read_csv(f"/home/coder/data-REU/{tab_files[i]}", separator="\t", has_header=False))
            value_df = pd.DataFrame(pl.read_csv(f"/home/coder/data-REU/{tab_files[j]}", separator="\t", has_header=False))

            # Stack key_df and value_df on top of each other.
            stacked_df = pd.concat([key_df, value_df], ignore_index=True)

            # Perform a natural sort on stacked_df on the chromosome, as well as the start and end locations.
            sorted_df = stacked_df.sort_values(by=[6, 7, 8], key=natsorted)

            # Get the total RNA signal from each location.
            signal_df = sorted_df.groupby(by=[6, 7, 8]).agg({9: "sum"}).round(5).reset_index()

            # Merge left on the sum of the RNA signals.
            res_df = pd.merge(signal_df, sorted_df, on=[6, 7, 8, 9], how="left")
            res_df = res_df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
            res_df.to_csv(f"/home/coder/REU/stranded_files/{cell_line}.stranded.{value_file_name}.{key_file_name}.coverage.txt", sep="\t", index=False, header=False)
