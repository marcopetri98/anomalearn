from __future__ import annotations

import abc

from ...abc import EqualityABC, FullyRepresentableABC
from .. import ICopyable


class IPipelineLayer(ICopyable, FullyRepresentableABC, EqualityABC):
    """The interface exposed from a layer of the pipeline.

    A layer of a pipeline must be copiable, savable, loadable and must have
    hyperparameters setters and getters (eventually empty if there are not).
    Individually, a pipeline layer must implement also at least one other
    interface. This interface is required for all layers that can be inserted in
    a pipeline object.
    """
    def __init__(self):
        super().__init__()
        
        self._pipeline_class = None
    
    def get_pipeline_class(self):
        """Gets the class to be used in the pipeline when the model has multiple allowed interfaces.
        
        Returns
        -------
        interface_to_use
            The interface to be used for the object.
        """
        return self._pipeline_class
    
    def set_pipeline_class(self, interface) -> IPipelineLayer:
        """Set which interface must be used by the pipeline.
        
        Parameters
        ----------
        interface
            One of the interfaces of the object that must be used in the
            pipeline, or None to reset it.

        Returns
        -------
        self
            Instance to itself
        """
        if interface is not None and not isinstance(self, interface):
            raise ValueError(f"self must be of type {interface}")
        
        self._pipeline_class = interface
        return self
    
    @abc.abstractmethod
    def get_input_shape(self) -> tuple:
        """Gets the input shape expected by the layer, eventually symbolic.

        Returns
        -------
        expected_input_shape : tuple
            It is the tuple representing the type of input shape that the layer
            expects as input. The tuple must be complete, considering all
            dimensions. If a dimension can be variable, it should be expressed
            with a string/letter, e.g., ("n", 5, 4) if the layer receives arrays
            with any dimensionality for axis 0 and dimension 5 and 5 for axis 1
            and 2. If two letters are identical, they represent the same value,
            e.g. ("n", "n") can be any array with two dimensions with equal
            value such as (5, 5) or (100, 100).
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_output_shape(self) -> tuple:
        """Gets the output shape expected by the layer, eventually symbolic.

        Returns
        -------
        expected_output_shape : tuple
            It is the tuple representing the type of output shape that the layer
            will emit. The tuple must be complete, considering all dimensions.
            If a dimension can be variable, it should be expressed with a
            string/letter, e.g., ("n", 5, 4) if the layer receives arrays with
            any dimensionality for axis 0 and dimension 5 and 5 for axis 1 and
            2. If two letters are identical, they represent the same value, e.g.
            ("n", "n") can be any array with two dimensions with equal value
            such as (5, 5) or (100, 100).
        """
        raise NotImplementedError
