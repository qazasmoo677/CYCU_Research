import pandas as pd
import numpy as np
from tqdm import tqdm

json_path = 'yelp_academic_dataset_review.json'
csv_name = "yelp_academic_dataset_review.csv"
df = pd.read_json(json_path, lines=True)
chunks = np.array_split(df.index, 100)
for ix, subset in tqdm(enumerate(chunks)):
    if ix == 0:
        df.loc[subset].to_csv(csv_name, mode='w', index=True)
    else:
        df.loc[subset].to_csv(csv_name, header=None, mode='a', index=True)