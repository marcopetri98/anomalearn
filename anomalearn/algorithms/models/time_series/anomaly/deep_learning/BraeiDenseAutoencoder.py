from typing import Tuple

import tensorflow as tf

from .TimeSeriesAnomalyAutoencoder import TimeSeriesAnomalyAutoencoder


class BraeiDenseAutoencoder(TimeSeriesAnomalyAutoencoder):
    """BraeiDenseAutoencoder"""
    
    def __init__(self, window: int = 200,
                 forecast: int = 1,
                 batch_size: int = 32,
                 max_epochs: int = 50,
                 predict_validation: float = 0.2,
                 batch_divide_training: bool = False,
                 folder_save_path: str = "data/nn_models/",
                 filename: str = "dense_ae",
                 extend_not_multiple: bool = True,
                 distribution: str = "gaussian",
                 perc_quantile: float = 0.999,
                 train_overlapping: bool = True,
                 test_overlapping: bool = True):
        super().__init__(window=window,
                         forecast=forecast,
                         batch_size=batch_size,
                         max_epochs=max_epochs,
                         predict_validation=predict_validation,
                         batch_divide_training=batch_divide_training,
                         folder_save_path=folder_save_path,
                         filename=filename,
                         extend_not_multiple=extend_not_multiple,
                         distribution=distribution,
                         perc_quantile=perc_quantile,
                         train_overlapping=train_overlapping,
                         test_overlapping=test_overlapping)
    
    def _prediction_create_model(self, input_shape: Tuple) -> tf.keras.Model:
        return self._learning_create_model(input_shape)
    
    def _learning_create_model(self, input_shape: Tuple) -> tf.keras.Model:
        input_layer = tf.keras.layers.Input(input_shape,
                                            name="input")
        
        flattening = tf.keras.layers.Flatten(name="flatten")(input_layer)
        
        encoder = tf.keras.layers.Dense(32,
                                        activation="relu",
                                        name="encoder_dense_1")(flattening)
        encoder = tf.keras.layers.Dense(16,
                                        activation="relu",
                                        name="encoder_dense_2")(encoder)
        
        latent = tf.keras.layers.Dense(8,
                                       activation="relu",
                                       name="latent_layer")(encoder)
        
        decoder = tf.keras.layers.Dense(16,
                                        activation="relu",
                                        name="decoder_dense_1")(latent)
        decoder = tf.keras.layers.Dense(32,
                                        activation="relu",
                                        name="decoder_dense_2")(decoder)
        
        output_layer = tf.keras.layers.Dense(input_shape[0] * input_shape[1],
                                             activation="linear",
                                             name="output")(decoder)
        
        model = tf.keras.Model(inputs=input_layer,
                               outputs=output_layer,
                               name="dense_autoencoder")
        
        model.compile(loss="mse",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanSquaredError()])
        
        return model
