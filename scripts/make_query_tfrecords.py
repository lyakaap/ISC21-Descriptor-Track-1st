import os
from pathlib import Path
import joblib

import tensorflow as tf
import pandas as pd
from tqdm import tqdm


def process_img(path):
    image = tf.io.read_file(str(path))

    image = tf.image.decode_jpeg(image)
    # image = tf.image.resize(image, size=(256, 256), method='nearest')
    image = tf.image.convert_image_dtype(image, tf.uint8)
    
    image = tf.image.encode_jpeg(image, quality=95, optimize_size=True)

    return image


out_dir = Path('/mnt/sdb/yokoo/fbisc/input/tfrecords/query')
out_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted(Path('/mnt/sdb/yokoo/fbisc/input/query_images').glob('**/*.jpg'))

public_ground_truth = pd.read_csv('/mnt/sdb/yokoo/fbisc/input/public_ground_truth.csv', index_col='query_id')
for path in image_paths:
    if path.stem not in public_ground_truth.index:
        public_ground_truth.loc[path.stem, 'reference_id'] = ''
public_ground_truth['reference_id'] = public_ground_truth['reference_id'].fillna('')

chunk_size = len(image_paths) // 128 + 1
chunked_image_paths = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]

for chunk_idx, chunk in tqdm(enumerate(chunked_image_paths), total=len(chunked_image_paths)):
    jobs = [joblib.delayed(process_img)(fp) for fp in chunk]
    bs = 10
    processed_images_chunk = joblib.Parallel(
        n_jobs=os.cpu_count(),
        verbose=0,
        require='sharedmem',
        batch_size=bs,
        backend='threading',
    )(jobs)

    with tf.io.TFRecordWriter(str(out_dir / f'chunk_{chunk_idx}.tfrecords')) as writer:
        for image, path in zip(processed_images_chunk, chunk):
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                'image_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(path.stem)])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
                'reference_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(public_ground_truth.loc[path.stem, 'reference_id'])])),
            })).SerializeToString()
            writer.write(record_bytes)
