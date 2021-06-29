import pickle
import random
import pandas as pd
gt = pd.read_csv('/home/shuhei.yokoo/fbisc/input/public_ground_truth.csv')

rids_from_gt = set(gt[gt.reference_id.notna()].reference_id.values)
random_ints = random.choices(range(1_000_000), k=1_000_00)
random_ids = set([f'R{id_:06d}' for id_ in random_ints])

rids_subset = rids_from_gt | random_ids
print(len(rids_subset))

with open('/home/shuhei.yokoo/fbisc/input/rids_subset.pickle', 'wb') as f:
    pickle.dump(list(rids_subset), f)
