import tensorflow as tf

def create_model(inputs, outputs):
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.train.GradientDescentOptimizer(0.001),
        loss='mse')
    return model
