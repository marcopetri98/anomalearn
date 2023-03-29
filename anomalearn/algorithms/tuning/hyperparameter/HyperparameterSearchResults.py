import logging

import numpy as np

from . import IHyperparameterSearchResults


class HyperparameterSearchResults(IHyperparameterSearchResults):
    """It stores the results of a hyperparameter search.
    
    Parameters
    ----------
    history : ndarray of shape (n_search + 1, params + 1)
        It represents the history of the hyperparameter search. The first row is
        the name of the parameters represented by the columns of the history.
    """
    
    def __init__(self, history: list):
        super().__init__()

        self.__logger = logging.getLogger(__name__)
        self._history = history
    
    def get_best_score(self) -> float:
        if len(self._history) == 0:
            self.__logger.info("There is no search")
            return np.nan
        else:
            return np.min(self._history[1:], axis=0)[0]
    
    def get_best_config(self) -> dict:
        if len(self._history) == 0:
            self.__logger.info("There is no search")
            return dict()
        else:
            tries = self._get_descending_score_tries()
            
            best_config = dict()
            for i, key in enumerate(tries[0]):
                best_config[key] = tries[-1][i]
            
            return best_config
    
    def get_num_iterations(self) -> int:
        return len(self._history) - 1
    
    def get_history(self) -> list:
        return self._history.copy()
    
    def print_search(self, *args, **kwargs) -> None:
        if len(self._history) == 0:
            print("There is no search")
        else:
            tries = self._get_descending_score_tries()
            
            print("Total number of tries: ", self.get_num_iterations())
            first = True
            for config in tries:
                if first:
                    first = False
                else:
                    text = ""
                    for i in range(len(config)):
                        text += str(tries[0][i])
                        text += ": " + str(config[i]) + ", "
                    print(text)
    
    def _get_descending_score_tries(self) -> list:
        """Gets all the tries in descending order (worst top, best bottom).
        
        Returns
        -------
        all_tries
            Tries ordered from worst to best.
        """
        scores = [x[0] for x in self._history[1::]]
        tries = np.array(scores)
        indices = (np.argsort(tries) + 1).tolist()
        indices.reverse()
        indices.insert(0, 0)
        tries = [self._history[i] for i in indices]
        return tries
