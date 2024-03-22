import pickle
import pandas as pd
import numpy as np
import jsonlines

path_block = './dev/blocks.pkl'

with open(path_block, 'rb') as f:
    df = pickle.load(f)

result = []
for _, row in df.iterrows():
    id, block, label = row['id'], row['code'], row['label']
    item = {'id': id, 'block': block, 'label': label}
    if len(block) > 30:
        result.append(item)

print(len(result))
with jsonlines.open('dev.jsonl', 'w') as writer:
    writer.write_all(result)


'''
percentiles = np.percentile(values, [90, 92, 94, 96, 98, 100])
for p, value in zip([90, 92, 94, 96, 98, 100], percentiles):
    print(f" {p}%: {value}")
'''
