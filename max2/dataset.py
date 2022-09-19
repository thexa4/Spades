
import tensorflow as tf
from functools import partial

def keys():
    fields = {
        'bid_state_bids': (4*15, tf.uint8),
        'bid_state_hand': (52, tf.uint8),
        'bid_state_bags': (2*10, tf.uint8)
    }
    for i in range(13):
        roundname = 'round' + str(i) + '_'
        fields[roundname + 'seen'] = (52, tf.uint8)
        fields[roundname + 'hand'] = (52, tf.uint8)
        fields[roundname + 'played'] = (3*52, tf.uint8)
        fields[roundname + 'todo'] = (4*26, tf.uint8)

    fields['chosen_bid'] = (1, tf.uint8)
    for i in range(13):
        roundname = 'round' + str(i) + '_'
        fields[roundname + 'card'] = (1, tf.uint8)
    fields['bid_result'] = (1, tf.float32)
    fields['rounds_result'] = (13, tf.float32)

    return fields
def size():
    return 4934

def encode(row_in, row_out):
    fielddefs = keys()
    elements = []
    for k in fielddefs.keys():
        v = fielddefs[k]
        ftype = v[1]
        fsize = v[0]

        if ftype == tf.uint8:
            elements.append(tf.cast(row_in[k], tf.uint8))
        elif ftype == tf.float32:
            casted = tf.bitcast(row_out[k], tf.uint8)
            flattened = tf.reshape(casted, (-1, 4 * fsize))
            elements.append(flattened)
        else:
            print('Bad type')
            exit(1)
            return
    
    return tf.concat(elements, -1)

def decode(is_bytes, raw):
    if is_bytes:
        arr = raw
    else:
        arr = tf.io.decode_raw(raw, tf.uint8)
    fielddefs = keys()
    inputs = {}
    outputs = {}
    pos = 0

    for k in fielddefs.keys():
        v = fielddefs[k]
        ftype = v[1]
        fsize = v[0]

        if ftype == tf.uint8:
            inputs[k] = tf.ensure_shape(arr[:, pos:(pos + fsize)], (None, fsize))
            pos += fsize
        elif ftype == tf.float32:
            b = arr[:, pos:(pos + fsize * 4)]
            pos += fsize * 4
            arranged = tf.reshape(b, (-1, fsize, 4))
            outputs[k] = tf.ensure_shape(tf.bitcast(arranged, tf.float32), (None, fsize))
        else:
            print('Bad type')
            exit(1)
            return
    
    return (inputs, outputs)

def load_raw(dataset, batchsize=64 * 1024):
    result = result.batch(batchsize)
    result = result.map(partial(decode, True))

    return result

def load(files, batchsize=64 * 1024):
    result = tf.data.FixedLengthRecordDataset(files, size(), num_parallel_reads=tf.data.AUTOTUNE, compression_type='GZIP')
    result = result.shuffle(256 * 1024)
    result = result.batch(batchsize)
    result = result.map(partial(decode, False))
    
    return result