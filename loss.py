import tensorflow as tf

class TripletLoss():
    def __init__(self, alpha=0.2):
        super(TripletLoss, self).__init__()
        """
        Initialize the Triplet Loss.

        Parameters:
        - alpha: The margin between positive and negative pairs.
        """
        self.alpha = alpha

    def __call__(self, anchor, positive, negative):
        """
        Compute the triplet loss.

        Parameters:
        - anchor: The anchor embedding.
        - positive: The positive embedding. 
        - negative: The negative embedding.

        Returns:
        - The triplet loss value.
        """

        # Compute the distance between anchor and positive
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)

        # Compute the distance between anchor and negative
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # Compute the triplett loss
        loss = tf.maximum(pos_dist - neg_dist + self.alpha, 0.0)

        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super(TripletLoss, self).get_config()
        config.update({
            'alpha': self.alpha,
        })
        return config


