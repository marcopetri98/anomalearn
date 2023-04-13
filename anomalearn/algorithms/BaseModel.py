class BaseModel:
    """Object representing a general model"""

    def get_hyperparameters(self, *args, **kwargs) -> dict:
        """Gets all the hyperparameters of the model and their allowed values.

        Parameters
        ----------
        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        hyperparameters : dict
            A dictionary with all the hyperparameters and the set of their
            possible values. Every key of the dictionary is the name of the
            hyperparameter. Each value of the dictionary is another dictionary
            with two keys: "value" and "set". The former key identifies the
            current value of the hyperparameter, the latter identifies the
            allowed values for the model. The allowed values for the model may
            be one of `Categorical`, `Integer` or `Real` classes from `skopt`.
        """
        return dict()

    def set_hyperparameters(self, hyperparameters: dict, *args, **kwargs) -> None:
        """Sets the hyperparameters of the model.

        Parameters
        ----------
        hyperparameters : dict
            A dictionary with all the hyperparameters values. Each key is the
            name of a hyperparameter and the value is the value assumed by the
            parameter.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        return

    def set_params(self, **params) -> None:
        """Modify the parameters of the object.

        Parameters
        ----------
        params
            The dictionary of the parameters to modify.

        Returns
        -------
        None
        """
        for key, value in params.items():
            if key not in self.__dict__:
                raise ValueError(f"Parameter '{key}' does not exist in class "
                                 f"'{self.__class__.__name__}'. Please, read "
                                 "either the signature or the docs for that "
                                 "class.")
            
            self.__dict__[key] = value

    def get_params(self, deep=True) -> dict:
        """Gets all the parameters (public attributes) of the model.

        Parameters
        ----------
        deep : bool, default=True
            States whether the method must return also the parameters of nested
            base models.

        Returns
        -------
        param_dict : dict
            Dictionary with parameters' names as keys and values as values.
        """
        all_parameters = self._get_all_params(deep=deep)
        public_parameters = {key: value
                             for key, value in all_parameters.items()
                             if not key.startswith("_")}
        return public_parameters

    def _get_all_params(self, deep=True) -> dict:
        """Gets all the parameters and attributes of the model.

        Parameters
        ----------
        deep : bool, default=True
            States whether the method must return also the parameters of nested
            base models.

        Returns
        -------
        param_dict : dict
            Dictionary with parameters' names as keys and values as values.
        """
        parameters = vars(self)
        params_to_return = dict()

        for key, value in parameters.items():
            if deep and isinstance(value, BaseModel):
                params_to_return[key] = value.get_params(deep)
            else:
                params_to_return[key] = value

        return params_to_return
