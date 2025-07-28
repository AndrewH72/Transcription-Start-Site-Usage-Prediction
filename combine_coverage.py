import os
import re
import pandas as pd
import numpy as np
import polars as pl
from tqdm import tqdm
from natsort import natsorted
from multiprocessing import Pool

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

# Splitting .tab and .txt files into different arrays.
# Sorting each array first by whether it is a plus or minus and then by its cell line.
data = os.listdir("data/RNASeq_bw")
# tab_files = sorted([file for file in data if file.endswith(".tab") and "unstranded" not in file], key=lambda x: (x[x.find(".") + 1], x[0]))
# coverage_files = sorted([file for file in data if file.endswith(".txt") and "unstranded" not in file], key=lambda x: (x[x.find(".") + 1], x[0]))

# def combine_strands(dict_item):
#     key, value = dict_item
#     key_df = pd.read_csv(f"data/RNASeq_bw/{key}", sep="\t", header=None)
#     value_df = pd.read_csv(f"data/RNASeq_bw/{value}", sep="\t", header=None)

#     # Stack key_df and value_df on top of each other.
#     stacked_df = pd.concat([key_df, value_df], ignore_index=True)

#     # Perform a natural sort on stacked_df on the chromosome, as well as the start and end locations.
#     sorted_df = stacked_df.sort_values(by=[6, 7, 8], key=natsorted)

#     # Get the total RNA signal from each location.
#     signal_df = sorted_df.groupby(by=[6, 7, 8]).agg({9: "sum"}).round(5).reset_index()

#     # Merge left on the sum of the RNA signals.
#     res_df = pd.merge(signal_df, sorted_df, on=[6, 7, 8, 9], how="left")
#     res_df = res_df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
#     return res_df

# if __name__ == "__main__":
#     file_pairs = {}
#     for file in coverage_files[:8]:
#         cell_line = file[ :file.find(".")]
#         sample_name = file[file.find("E"): file.find("E") + 11]
#         for file2 in coverage_files[8:-8]:
#             sample_name2 = file2[file2.find("E"): file2.find("E") + 11]
#             if strand_pairs[sample_name] == sample_name2:
#                 file_pairs[file] = file2

#     pool = Pool(processes=4)
#     for res_df in tqdm(pool.imap(combine_strands, file_pairs.items()), total=len(file_pairs)):
#         res_df.to_csv(f"data/RNASeq_bw{cell_line}.stranded.{sample_name}.{sample_name2}.coverage.txt", sep="\t", index=False)