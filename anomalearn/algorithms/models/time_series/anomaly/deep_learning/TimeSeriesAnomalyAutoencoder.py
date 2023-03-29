from abc import ABC
from typing import Tuple

import numpy as np
from keras.callbacks import History
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from .TimeSeriesAnomalyWindowDL import TimeSeriesAnomalyWindowDL
from ......utils.printing import print_warning


class TimeSeriesAnomalyAutoencoder(TimeSeriesAnomalyWindowDL, ABC):
    """A framework model for autoencoders for anomaly detection
    
    Parameters
    ----------
    extend_not_multiple : bool, default=True
        States whether the training and validation sets must be resized in case
        they are not a multiple of window. If true, exceeding elements will be
        discarded to train and validate on multiple of window size.
        
    train_overlapping : bool, default=True
        States whether the training examples can overlap. An autoencoder takes a
        window and projects it onto the latent space and back-projects it to the
        starting space. For time series we take 1,2,...,w points as window where
        w is window length, and we project them onto latent space. If this flag
        is True, we define for training examples such as 1,2,...,w and overlapped
        ones such as 2,3,...,w+1 and 3,4,...,w+2. If it is false, we will
        generate 1,2,...,w, w+1, w+2,...,2w and so on.
        
    test_overlapping : bool, default=True
        States whether the test examples can overlap. Overlapping works exactly
        as in the training. Some points will be predicted from at most ```window```
        windows. If overlapping on test is set to true, each point will be
        predicted as the average value of all the windows predicting that point.
    """
    
    def __init__(self, window: int = 200,
                 forecast: int = 1,
                 batch_size: int = 32,
                 max_epochs: int = 50,
                 predict_validation: float = 0.2,
                 batch_divide_training: bool = False,
                 folder_save_path: str = "data/nn_models/",
                 filename: str = "autoencoder",
                 extend_not_multiple: bool = True,
                 distribution: str = "gaussian",
                 perc_quantile: float = 0.999,
                 train_overlapping: bool = True,
                 test_overlapping: bool = True):
        super().__init__(window=window,
                         stride=window,
                         forecast=forecast,
                         batch_size=batch_size,
                         max_epochs=max_epochs,
                         predict_validation=predict_validation,
                         batch_divide_training=batch_divide_training,
                         folder_save_path=folder_save_path,
                         filename=filename,
                         distribution=distribution,
                         perc_quantile=perc_quantile)
        
        self.extend_not_multiple = extend_not_multiple
        self.train_overlapping = train_overlapping
        self.test_overlapping = test_overlapping
    
    def fit(self, x, training_idx, validation_idx, y) -> list[History]:
        check_X_y(x, y)
        x = np.array(x)
        y = np.array(y)
        
        if self.extend_not_multiple:
            print_warning("Data will be modified to match the window size, note"
                          " that the input must be a multiple of the window "
                          "size for autoencoders. This is just a comfort "
                          "utility. You can ignore this warning if no other "
                          "warnings regarding the training set pop up.")
            
            for i in range(len(training_idx)):
                slice_ = training_idx[i]
                if (slice_.stop - slice_.start) % self.window != 0:
                    remainder = (slice_.stop - slice_.start) % self.window
                    training_idx[i] = slice(slice_.start, slice_.stop - remainder)
                    
                    print_warning("On ", i+1, "th training slice, ", remainder,
                                  " points have been lost")
            
            for i in range(len(validation_idx)):
                slice_ = validation_idx[i]
                if (slice_.stop - slice_.start) % self.window != 0:
                    remainder = (slice_.stop - slice_.start) % self.window
                    validation_idx[i] = slice(slice_.start, slice_.stop - remainder)
                    
                    print_warning("On ", i+1, "th validation slice, ",
                                  remainder, " points have been lost")
        
        history = super().fit(x, training_idx, validation_idx, y)
        return history
    
    def _build_x_y_sequences(self, x) -> Tuple[np.ndarray, np.ndarray]:
        samples = []
        targets = []
        
        if not self.train_overlapping:
            increment = self.stride
        else:
            increment = 1
        
        for i in range(0, x.shape[0] - self.window, increment):
            samples.append(x[i:i + self.window])
            targets.append(x[i:i + self.window])
        
        return np.array(samples), np.array(targets)
    
    def _predict_future(self, xp: np.ndarray, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        check_array(x)
        x = np.array(x)
        
        if not self.test_overlapping and x.shape[0] % self.window != 0:
            raise ValueError("The autoencoder can predict only a number of "
                             "points such that points % self.window == 0")
        
        if not self.test_overlapping:
            increment = self.stride
        else:
            increment = 1
        
        # I build the array of inputs for the model
        inputs = []
        window_mask = np.zeros(x.shape[0])
        for i in range(0, x.shape[0] - self.window + 1, increment):
            next_input = x[i:i + self.window]
            window_mask[i:i + self.window] += 1
            inputs.append(next_input.copy())
        inputs = np.array(inputs)
        
        predictions = self.model_.predict(inputs, batch_size=1)
        
        # Compute the average on the overlapping windows
        final_predictions = np.zeros(x.shape[0])
        predictions = predictions.reshape((predictions.shape[0],
                                           predictions.shape[1]))
        for i in range(predictions.shape[0]):
            start = i*increment
            final_predictions[start:start + self.window] += predictions[i]
        final_predictions = final_predictions / window_mask
        final_predictions = final_predictions.reshape(x.shape[0], 1)
        
        return final_predictions
    
    def predict_time_series(self, xp, x) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        check_array(x)
        x = np.array(x)
        
        if x.shape[0] < self.window:
            raise ValueError("You must provide at lest window points to predict")
        elif not self.test_overlapping and x.shape[0] % self.window != 0:
            raise ValueError("An autoencoder must receive as input a multiple "
                             "of window to be able to predict. Namely, the input"
                             "must be such that X.shape[0] % self.window == 0")
        
        return self._predict_future(xp, x)
