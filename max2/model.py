
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def create():
    inputs = {}
    bid_state_bids = keras.Input(shape=(4*15), name='bid_state_bids')
    inputs['bid_state_bids'] = bid_state_bids
    inputs['bid_state_hand'] = keras.Input(shape=(52,), name='bid_state_hand')
    inputs['bid_state_bags'] = keras.Input(shape=(2*10), name='bid_state_bags')
    bid_state = layers.Concatenate(name='bid_state')([layers.Reshape(target_shape=(60,))(bid_state_bids), inputs['bid_state_hand'], layers.Reshape(target_shape=(20,))(inputs['bid_state_bags'])])

    rounds = []
    for i in range(13):
        roundname='round' + str(i) + '_'
        seen = keras.Input(shape=(52,), name=roundname + 'seen')
        hand = keras.Input(shape=(52,), name=roundname + 'hand')
        played = keras.Input(shape=(3* 52), name=roundname + 'played')
        played_flat = layers.Flatten()(played)
        todo = keras.Input(shape=(4*26), name=roundname + 'todo')
        todo_flat = layers.Flatten()(todo)
        inputs[roundname + 'seen'] = seen
        inputs[roundname + 'hand'] = hand
        inputs[roundname + 'played'] = played
        inputs[roundname + 'todo'] = todo
        concat = layers.Concatenate(name='round' + str(i))([seen, hand, played_flat, todo_flat, bid_state])
        rounds.append(layers.Reshape(target_shape=([1, concat.shape[1]]))(concat))
        
    training_inputs = {}
    training_rounds = []
    training_inputs['chosen_bid'] = keras.Input(shape=(1,), dtype='int8', name='chosen_bid')
    for i in range(13):
        roundname='round' + str(i) + '_'
        chosen_card = keras.Input(shape=(1,), dtype='int8', name=roundname + 'card')
        training_rounds.append(layers.Reshape(target_shape=(1,1))(chosen_card))
        training_inputs[roundname + 'card'] = chosen_card
    training_rounds = layers.Reshape(target_shape=(13,))(layers.Concatenate(axis=1)(training_rounds))

    bid_hidden = layers.Dense(32, activation='tanh')(bid_state)
    bid_output = layers.Lambda(lambda x: x * 200)(layers.Dense(14, activation='tanh')(bid_hidden))
        
    rounds_stacked = layers.Concatenate(axis=1)(rounds)
    hidden_lstm = layers.LSTM(32, return_sequences=True)(rounds_stacked)
    ltsm = layers.Lambda(lambda x: x * 200)(layers.LSTM(52, return_sequences=True)(hidden_lstm))

    inference_model = keras.Model(inputs=inputs, outputs={'bid_result': bid_output, 'rounds_result': ltsm}, name="spades1")

    bid_mask = layers.Reshape(target_shape=(14,))(OneHot(14, 1)(training_inputs['chosen_bid']))
    rounds_mask = OneHot(52, 13)(training_rounds)
    bid_result = layers.Dot(name='bid_score', axes=1)([bid_output, bid_mask])
    rounds_result = layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.multiply(x[0], x[1]), axis=2), name='rounds_result')([ltsm, rounds_mask])

    training_model = keras.Model(inputs={**inputs, **training_inputs}, outputs={'bid_result': bid_result, 'rounds_result': rounds_result}, name="spades1")

    return (inference_model, training_model)

def load(q, generation):
    interpreter = tf.lite.Interpreter(model_path=f'max2/models/q{q}/gen{generation:03}.tflite')
    return interpreter.get_signature_runner()
    #return tf.keras.models.load_model(f'max2/models/q{q}/gen{generation:03}.model', compile=False)