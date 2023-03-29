from typing import Tuple

import tensorflow as tf

from .TimeSeriesAnomalyAutoencoder import TimeSeriesAnomalyAutoencoder


class CNNAutoencoder(TimeSeriesAnomalyAutoencoder):
    """CNNAutoencoder"""
    
    def __init__(self, window: int = 200,
                 forecast: int = 1,
                 batch_size: int = 32,
                 max_epochs: int = 50,
                 predict_validation: float = 0.2,
                 batch_divide_training: bool = False,
                 folder_save_path: str = "data/nn_models/",
                 filename: str = "cnn_ae",
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
        
        encoder = tf.keras.layers.Conv1D(16,
                                         3,
                                         padding="same",
                                         activation="relu",
                                         name="encoder_cnn_1")(input_layer)
        encoder = tf.keras.layers.MaxPool1D(2,
                                            name="encoder_pool_1")(encoder)
        encoder = tf.keras.layers.Conv1D(8,
                                         3,
                                         padding="same",
                                         activation="relu",
                                         name="encoder_cnn_2")(encoder)
        encoder = tf.keras.layers.MaxPool1D(2,
                                            name="encoder_pool_2")(encoder)
        encoder = tf.keras.layers.Conv1D(8,
                                         3,
                                         padding="same",
                                         activation="relu",
                                         name="encoder_cnn_3")(encoder)
        
        decoder = tf.keras.layers.Conv1DTranspose(8,
                                                  3,
                                                  padding="same",
                                                  activation="relu",
                                                  name="decoder_cnn_1")(encoder)
        decoder = tf.keras.layers.UpSampling1D(2,
                                               name="decoder_up_1")(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(8,
                                                  3,
                                                  padding="same",
                                                  activation="relu",
                                                  name="decoder_cnn_2")(decoder)
        decoder = tf.keras.layers.UpSampling1D(2,
                                               name="decoder_up_2")(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(16,
                                                  3,
                                                  padding="same",
                                                  activation="relu",
                                                  name="decoder_cnn_3")(decoder)
        
        output_layer = tf.keras.layers.Conv1D(1,
                                              3,
                                              padding="same",
                                              activation="linear")(decoder)
        
        model = tf.keras.Model(inputs=input_layer,
                               outputs=output_layer,
                               name="cnn_autoencoder")
        
        model.compile(loss="mse",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanSquaredError()])
        
        return model
