
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return keras.backend.one_hot(keras.backend.cast(x, 'uint8'),
                                     num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return layers.Lambda(_one_hot,
                         arguments={'num_classes': input_dim},
                         input_shape=(input_length,))


def to_bid(layer, name=None):
    return layers.Lambda(lambda x: keras.backend.clip(x * 360, -360, 360), name=name)(layer)


def create(scale=32):
    inputs = {}
    bid_state_bids = keras.Input(shape=(4*15), name='bid_state_bids')
    inputs['bid_state_bids'] = bid_state_bids
    inputs['bid_state_hand'] = keras.Input(shape=(52,), name='bid_state_hand')
    inputs['bid_state_bags'] = keras.Input(shape=(2*10), name='bid_state_bags')
    bid_state = layers.Concatenate(name='bid_state')([layers.Reshape(target_shape=(60,))(
        bid_state_bids), inputs['bid_state_hand'], layers.Reshape(target_shape=(20,))(inputs['bid_state_bags'])])

    bid_hidden = layers.Dense(2 * scale, activation='relu')(bid_state)
    bid_hidden = layers.Dense(2 * scale, activation='relu')(bid_hidden)
    bid_hidden = layers.Dense(2 * scale, activation='relu')(bid_hidden)
    bid_hidden = layers.Dense(2 * scale, activation='relu')(bid_hidden)

    sliding_layer1 = layers.Dense(4 * scale, activation='relu')
    sliding_layer2 = layers.Dense(4 * scale, activation='relu')
    sliding_layer3 = layers.Dense(4 * scale, activation='relu')
    rounds = []
    for i in range(13):
        roundname = 'round' + str(i) + '_'
        seen = keras.Input(shape=(52,), name=roundname + 'seen')
        hand = keras.Input(shape=(52,), name=roundname + 'hand')
        played = keras.Input(shape=(3 * 52), name=roundname + 'played')
        played_flat = layers.Flatten()(played)
        todo = keras.Input(shape=(4*26), name=roundname + 'todo')
        todo_flat = layers.Flatten()(todo)
        inputs[roundname + 'seen'] = seen
        inputs[roundname + 'hand'] = hand
        inputs[roundname + 'played'] = played
        inputs[roundname + 'todo'] = todo
        concat = layers.Concatenate(
            name='round' + str(i))([seen, hand, played_flat, todo_flat, bid_hidden])
        cur_layer = layers.Reshape(target_shape=([1, concat.shape[1]]))(concat)
        rounds.append(sliding_layer3(
            sliding_layer2(sliding_layer1(cur_layer))))

    training_inputs = {}
    training_rounds = []
    training_inputs['chosen_bid'] = keras.Input(
        shape=(1,), dtype='int8', name='chosen_bid')
    for i in range(13):
        roundname = 'round' + str(i) + '_'
        chosen_card = keras.Input(
            shape=(1,), dtype='int8', name=roundname + 'card')
        training_rounds.append(layers.Reshape(
            target_shape=(1, 1))(chosen_card))
        training_inputs[roundname + 'card'] = chosen_card
    training_rounds = layers.Reshape(target_shape=(13,))(
        layers.Concatenate(axis=1)(training_rounds))

    bid_output = to_bid(layers.Dense(14, activation='tanh')
                        (bid_hidden), name='bid_output')

    rounds_stacked = layers.Concatenate(axis=1)(rounds)
    ltsm = layers.LSTM(4 * scale, activation='tanh',
                       return_sequences=True)(rounds_stacked)
    perlayer_common = layers.Dense(2 * scale, activation='relu')
    perlayer = []
    for i in range(13):
        last = ltsm[:, i, :]
        last = perlayer_common(last)
        last = layers.Dense(2 * scale, activation='relu')(last)
        last = layers.Dense(2 * scale, activation='relu')(last)
        perlayer.append(layers.Dense(52, activation='tanh')(last))
    rounds_result = to_bid(layers.Lambda(lambda x: tf.stack(
        x, axis=1))(perlayer), name='rounds_result')

    inference_model = keras.Model(inputs=inputs, outputs={
                                  'bid_result': bid_output, 'rounds_result': rounds_result}, name="spades1")

    bid_mask = layers.Reshape(target_shape=(14,))(
        OneHot(14, 1)(training_inputs['chosen_bid']))
    rounds_mask = OneHot(52, 13)(training_rounds)
    bid_result = layers.Dot(name='bid_score', axes=1)([bid_output, bid_mask])
    rounds_result = layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.multiply(
        x[0], x[1]), axis=2), name='rounds_result_training')([rounds_result, rounds_mask])

    training_model = keras.Model(inputs={**inputs, **training_inputs}, outputs={
                                 'bid_result': bid_result, 'rounds_result': rounds_result}, name="spades1")

    return (inference_model, training_model)


def create_v2(scale=32, depth=3):
    inputs = {}
    bid_state_bids = keras.Input(shape=(4*15), name='bid_state_bids')
    inputs['bid_state_bids'] = bid_state_bids
    inputs['bid_state_hand'] = keras.Input(shape=(52,), name='bid_state_hand')
    inputs['bid_state_bags'] = keras.Input(shape=(2*10), name='bid_state_bags')
    bid_state = layers.Concatenate(name='bid_state')([layers.Reshape(target_shape=(60,))(
        bid_state_bids), inputs['bid_state_hand'], layers.Reshape(target_shape=(20,))(inputs['bid_state_bags'])])

    bid_hidden = layers.Dense(4 * scale, activation='relu')(bid_state)
    for i in range(depth - 1):
        bid_hidden = layers.Dense(scale, activation='relu')(bid_hidden)

    current = bid_hidden
    layer_outputs = []

    for i in range(13):
        roundname = 'round' + str(i) + '_'
        seen = keras.Input(shape=(52,), name=roundname + 'seen')
        hand = keras.Input(shape=(52,), name=roundname + 'hand')
        played = keras.Input(shape=(3 * 52), name=roundname + 'played')
        played_flat = layers.Flatten()(played)
        todo = keras.Input(shape=(4*26), name=roundname + 'todo')
        todo_flat = layers.Flatten()(todo)
        inputs[roundname + 'seen'] = seen
        inputs[roundname + 'hand'] = hand
        inputs[roundname + 'played'] = played
        inputs[roundname + 'todo'] = todo
        concat = layers.Concatenate(
            name='round' + str(i))([seen, hand, played_flat, todo_flat, layers.Flatten()(current)])
        flattened = layers.Reshape(target_shape=([1, concat.shape[1]]))(concat)
        current = layers.Dense(4 * scale, activation='relu')(flattened)
        for i in range(depth - 1):
            current = layers.Dense(scale, activation='relu')(current)
        layer_outputs.append(layers.Dense(52, activation='tanh')(current))

    bid_output = to_bid(layers.Dense(14, activation='tanh')
                        (bid_hidden), name='bid_output')
    rounds_result = to_bid(layers.Lambda(lambda x: tf.stack(
        x, axis=1))(layer_outputs), name='rounds_result')
    rounds_result = layers.Reshape(target_shape=(13, 52))(rounds_result)

    training_inputs = {}
    training_rounds = []
    training_inputs['chosen_bid'] = keras.Input(
        shape=(1,), dtype='int8', name='chosen_bid')
    for i in range(13):
        roundname = 'round' + str(i) + '_'
        chosen_card = keras.Input(
            shape=(1,), dtype='int8', name=roundname + 'card')
        training_rounds.append(layers.Reshape(
            target_shape=(1, 1))(chosen_card))
        training_inputs[roundname + 'card'] = chosen_card
    training_rounds = layers.Reshape(target_shape=(13,))(
        layers.Concatenate(axis=1)(training_rounds))

    inference_model = keras.Model(inputs=inputs, outputs={
                                  'bid_result': bid_output, 'rounds_result': rounds_result}, name="spades1")

    bid_mask = layers.Reshape(target_shape=(14,))(
        OneHot(14, 1)(training_inputs['chosen_bid']))
    rounds_mask = OneHot(52, 13)(training_rounds)
    bid_result = layers.Dot(name='bid_score', axes=1)([bid_output, bid_mask])
    rounds_training_result = layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.multiply(
        x[0], x[1]), axis=2), name='rounds_result_training')([rounds_result, rounds_mask])

    training_model = keras.Model(inputs={**inputs, **training_inputs}, outputs={
                                 'bid_result': bid_result, 'rounds_result': rounds_training_result}, name="spades1")

    return (inference_model, training_model)


def execute(interpreter, data):
    tensors = interpreter.get_input_details()
    #print([x['name'] for x in interpreter.get_input_details()])
    for key in data.keys():
        for tensor in tensors:
            if key in tensor['name']:
                interpreter.set_tensor(tensor['index'], data[key])
                break
    interpreter.invoke()

    out = interpreter.get_output_details()

    smallerindex = 0
    if len(out[smallerindex]['shape']) != 2:
        smallerindex = 1

    return {
        'lambda': interpreter.get_tensor(out[smallerindex]['index']),
        'lambda_1': interpreter.get_tensor(out[1 - smallerindex]['index'])
    }
    # print(out)
    # print(data)
    # quit()


def load(q, generation):
    interpreter = tf.lite.Interpreter(
        model_path=f'max2/models/q{q}/gen{generation:03}.tflite')
    interpreter.allocate_tensors()
    #print([x['name'] for x in interpreter.get_input_details()])
    # print(interpreter.get_output_details())
    return lambda **x: execute(interpreter, x)

    # return interpreter.get_signature_runner()
    # return tf.keras.models.load_model(f'max2/models/q{q}/gen{generation:03}.model', compile=False)


def loadraw(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    return lambda **x: execute(interpreter, x)
