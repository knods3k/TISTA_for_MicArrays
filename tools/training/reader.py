import tensorflow as tf

def parse(record):
    feature_dict= { "y": tf.io.VarLenFeature(tf.float32),
                    "x": tf.io.VarLenFeature(tf.float32)}
    parse_ex    = lambda rec: tf.io.parse_single_example(rec, feature_dict)
    extract     = lambda example,key: tf.sparse.to_dense(example[key])
    y           = extract(parse_ex(record),"y")
    x           = extract(parse_ex(record),"x")
    return y, x 

def read(tfrecord):
    ignore_order= tf.data.Options()
    ignore_order.experimental_deterministic = False 

    raw_data    = tf.data.TFRecordDataset(tfrecord)
    raw_data    = raw_data.with_options(ignore_order)

    dataset     = raw_data.map(parse)
    return dataset

def batch(dataset,batchsize):
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(-1)
    dataset = dataset.batch(batchsize)
    return dataset

def get_batch(tfrecord,batchsize):
    return batch(read(tfrecord),batchsize)