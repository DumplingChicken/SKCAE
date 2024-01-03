import tensorflow as tf
from keras import backend as K

class Linear(tf.keras.layers.Layer):
    """
    the End-to-end mapping between two independent time steps
    """

    def __init__(self, **kwargs):
        super(Linear, self).__init__()

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or K.floatx())
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        init = tf.keras.initializers.GlorotUniform()
        # init = tf.keras.initializers.identity()
        
        self.op = tf.Variable(
            initial_value=init(shape=(last_dim, last_dim), dtype=self.dtype), 
            trainable=True, name='operator'
        )
            
    def call(self, inputs):
        self.w = tf.matmul(inputs, self.op)
        return self.w
        
    def get_weights(self):
        weight = []
        weight.append(self.op.numpy())
        return weight
    
    def get_w(self):
        weight = []
        weight.append(self.w.numpy())
        return weight
    
    def get_config(self):
        config = super(Linear, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)