import tensorflow as tf

class Model:

    """
    features:
     - hand
     - seen
     - bids
     - tricks
     - scores
     - bags
     - suits_empty
    """
    def create_network(features, mode, params):

        batch_size = tf.shape(features['bids'])[0]
        print(features['bids'])
        bids_hot = tf.reshape(tf.one_hot(
            indices = features['bids'],
            depth = 15,
            dtype = tf.float32,
            name = 'bids_hot',
        ), shape = [batch_size, 15 * 4], name='bids_hot_reshape')

        tricks_hot = tf.reshape(tf.one_hot(
            indices = features['tricks'],
            depth = 14,
            dtype = tf.float32,
            name = 'tricks_hot',
        ), shape = [batch_size, 14 * 4], name='tricks_hot_reshape')

        bags_hot = tf.reshape(tf.one_hot(
            indices = features['bags'],
            depth = 10,
            dtype = tf.float32,
            name = 'bags_hot',
        ), shape = [batch_size, 10 * 2], name = 'bags_hot_reshape')
        

        inputs = tf.concat([ tf.to_float(features['hand']), tf.to_float(features['seen']), bids_hot, tricks_hot, bags_hot, tf.to_float(features['suits_empty']) ], axis = -1)
        input_layer = tf.reshape(inputs, shape = [batch_size, 15 * 4 + 14 * 4 + 10 * 2 + 52 + 52 + 16])

        hidden_layer = tf.layers.dense(input_layer, 12, activation = tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(params['regularizer']), use_bias=True)

        dropout = tf.layers.dropout(hidden_layer, training = (mode == tf.estimator.ModeKeys.TRAIN), rate=0.5)

        score_output =  tf.multiply(tf.layers.dense(dropout, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(params['regularizer']), use_bias=True, activation=tf.tanh), 300.0)
        win_output =  tf.layers.dense(dropout, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(params['regularizer']), use_bias=True, activation=tf.sigmoid)

        return (score_output, win_output)


    def create_predictor(features, mode, params):

        score_output, win_output = Model.create_network(features, mode, params)
        
        predictions_dict = {
            "number": score_output,
            "win": win_output,
        }

        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions_dict,
            loss = tf.losses.get_regularization_loss(),
        )

    def create_trainer(features, labels, mode, params):
        score_output, win_output = Model.create_network(features, mode, params)

        tf.summary.histogram("number/guessed", score_output)
        tf.summary.histogram("number/diff", tf.subtract(score_output, tf.to_float(labels["score_delta"])))
        tf.summary.histogram("win/guessed", win_output)
        tf.summary.histogram("win/diff", tf.subtract(win_output, tf.to_float(labels["win_chance"])))

        loss = 0
        
        reg_loss = tf.losses.get_regularization_loss()
        loss += reg_loss
        tf.summary.scalar("loss/regularization", reg_loss)

        number_target = tf.reshape(labels["score_delta"], tf.shape(score_output))
        win_target = tf.reshape(labels["win_chance"], tf.shape(win_output))
        num_loss = tf.losses.mean_squared_error(number_target, score_output) * params['numbers_weight']
        win_loss = tf.losses.mean_squared_error(win_target, win_output) * params['win_weight']
        loss += num_loss
        loss += win_loss
        tf.summary.scalar("loss/numbers", num_loss)
        tf.summary.scalar("loss/win", win_loss)

        predictions_dict = {
            "number": score_output,
            "win": win_output,
        }

        tf.summary.scalar("loss/sum", loss)
        eval_metric_ops = {
            #"loss/numbers": num_loss,
            #"loss/regularization": reg_loss,
            #"loss/sum": loss,
        }

        train_op = tf.contrib.layers.optimize_loss(
            loss = loss,
            global_step = tf.train.get_global_step(),
            learning_rate = params['learnrate'],
            optimizer = "Adagrad",
        )

        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions_dict,
            loss = loss,
            train_op = train_op,
            eval_metric_ops = eval_metric_ops,
        )
