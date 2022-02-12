import jax
from jax import random

def _generate_hands(batch_size):
    deck = tf.map_fn(tf.random.shuffle, tf.tile(tf.reshape(tf.range(52), (1, 52)), [batch_size, 1]), name="deck")
    hand_indices = [deck[:,0:13], deck[:, 13:26], deck[:, 26:39], deck[:, 39:52]]

    hands = [
        _indices_to_hand(hand_indices[0]),
        _indices_to_hand(hand_indices[1]),
        _indices_to_hand(hand_indices[2]),
        _indices_to_hand(hand_indices[3]),
    ]

    scores = tf.random.uniform((2, batch_size), minval=-499, maxval=500, dtype=tf.int32)
    deltas = [
        scores[0] - scores[1],
        scores[1] - scores[0],
    ]
    can_blind = tf.less(deltas, -100)



    return (scores, can_blind)

def _randint(key, minval, maxval, shape=()):
    newkey, subkey = random.split(key)
    return (newkey, random.randint(subkey, shape, minval, maxval))

def game_simulator(player1, player2, player3, player4, score = None, rndkey = None):
    if rndkey == None:
        rndkey = random.PRNGKey(0)

    players = [player1, player2, player3, player4]
    rndkey, rotation = _randint(rndkey, 0, 4)
    players = players[rotation:] + players[:rotation]

    #hands = _generate_hands(batch_size)

    return (rndkey, players)

def main():
    print(game_simulator(10, 20, 30, 40))

if __name__=='__main__':
    main()
