from typing import Tuple

import tensorflow as tf

from .TimeSeriesAnomalySliding import TimeSeriesAnomalySliding


class BraeiLSTM(TimeSeriesAnomalySliding):
    """LSTM model to identify anomalies in time series."""
    
    def __init__(self, window: int = 200,
                 stride: int = 1,
                 forecast: int = 1,
                 batch_size: int = 32,
                 max_epochs: int = 50,
                 predict_validation: float = 0.2,
                 batch_divide_training: bool = False,
                 folder_save_path: str = "data/nn_models/",
                 filename: str = "lstm",
                 distribution: str = "gaussian",
                 perc_quantile: float = 0.999):
        super().__init__(window=window,
                         stride=stride,
                         forecast=forecast,
                         batch_size=batch_size,
                         max_epochs=max_epochs,
                         predict_validation=predict_validation,
                         batch_divide_training=batch_divide_training,
                         folder_save_path=folder_save_path,
                         filename=filename,
                         distribution=distribution,
                         perc_quantile=perc_quantile)
    
    def _prediction_create_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Creates the LSTM model to perform the predictions.
        
        This function should be used in case the prediction is to be performed
        with batch=1 while the training is done with fixed batch and a batch
        greater than 1 (the prediction batch). If the batch is not fixed in
        training, this function should call ``_learning_create_model``.

        Parameters
        ----------


        Returns
        -------
        model : tf.keras.Model
            The model for the prediction.
        """
        
        input_layer = tf.keras.layers.Input(input_shape,
                                            name="input",
                                            batch_size=1)
        
        lstm_1 = tf.keras.layers.LSTM(4,
                                      stateful=True,
                                      return_sequences=True,
                                      name="stateful_lstm_1")(input_layer)
        
        lstm_2 = tf.keras.layers.LSTM(4,
                                      stateful=True,
                                      name="stateful_lstm_2")(lstm_1)
        
        output_layer = tf.keras.layers.Dense(self.forecast * input_shape[1],
                                             name="output")(lstm_2)
        
        model = tf.keras.Model(inputs=input_layer,
                               outputs=output_layer,
                               name="lstm_model")
        
        model.compile(loss="mse",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanSquaredError()])
        
        return model
    
    def _learning_create_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Creates the LSTM model to perform the training.

        Returns
        -------
        model : tf.keras.Model
            The model for the prediction.
        """
        
        input_layer = tf.keras.layers.Input(input_shape,
                                            name="input",
                                            batch_size=self.batch_size)
        
        lstm_1 = tf.keras.layers.LSTM(4,
                                      stateful=True,
                                      return_sequences=True,
                                      name="stateful_lstm_1")(input_layer)
        
        lstm_2 = tf.keras.layers.LSTM(4,
                                      stateful=True,
                                      name="stateful_lstm_2")(lstm_1)
        
        output_layer = tf.keras.layers.Dense(self.forecast * input_shape[1],
                                             name="output")(lstm_2)
        
        model = tf.keras.Model(inputs=input_layer,
                               outputs=output_layer,
                               name="lstm_model")
        
        model.compile(loss="mse",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanSquaredError()])
        
        return model