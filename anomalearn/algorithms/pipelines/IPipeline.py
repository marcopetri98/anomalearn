import abc
from typing import Tuple

import numpy as np

from . import AbstractPipelineSavableLayer
from .. import IParametric, BaseModel


class IPipeline(IParametric, AbstractPipelineSavableLayer):
    """The interface for pipeline objects.
    """
    @abc.abstractmethod
    def allowed_interfaces(self) -> list:
        """Gets the list of allowed interfaces in the pipeline.
        
        Returns
        -------
        allowed_interfaces : list
            The list of the interface classes allowed in the pipeline.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def identical(self, other, degree: int = 2) -> bool:
        """Checks if the two pipelines are identical.

        Parameters
        ----------
        other
            Object to compare with.

        degree : [1, 2, 3], default=2
            Degree at which objects must be identical. With 1 it is identical
            to `==` operator. With 2 the layers' models must be equal and also
            the train flag. With 3 the two pipelines must have the same name,
            model and train flag for layers.

        Returns
        -------
        are_identical : bool
            States if the two pipelines are identical to degree `degree`.
            
        Raises
        ------
        TypeError
            If degree is not int.
            
        ValueError
            If degree is not 1, 2 or 3.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_trainable(self, layer: int | str, trainable: bool) -> None:
        """Sets the trainable flag for the specified layer.
        
        Parameters
        ----------
        layer : int or str
            The position of the layer in the pipeline or its unique name.
        
        trainable : bool
            The trainable flag to set.

        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If layer is neither int nor str or if trainable is not bool.
            
        IndexError
            If the integer or string passed to identify the layer is out of
            bounds or there are no layers with that name.
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def set_name(self, layer: int | str, name: str) -> None:
        """Sets a new name for the specified layer.
        
        Parameters
        ----------
        layer : int or str
            The position of the layer in the pipeline or its unique name.
        
        name : str
            The new name of the layer.

        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If layer is neither int nor str or if name is not str.
            
        IndexError
            If the integer or string passed to identify the layer is out of
            bounds or there are no layers with that name.
            
        ValueError
            If the name is already used by another layer.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def add_first_layer(self, layer_spec: Tuple[str, BaseModel, bool] | Tuple[str, BaseModel] | Tuple[BaseModel, bool] | BaseModel) -> None:
        """Adds the layer as first layer.

        Parameters
        ----------
        layer_spec : Tuple[str, BaseModel, bool] | Tuple[str, BaseModel] | Tuple[BaseModel, bool] | BaseModel
            A layer specified as it is specified in the init of the pipeline as
            a list element.

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the tuple length is greater than 3 or if the object passed is
            neither tuple nor BaseModel.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def insert_layer(self, index: int,
                     layer_spec: Tuple[str, BaseModel, bool] | Tuple[str, BaseModel] | Tuple[BaseModel, bool] | BaseModel) -> None:
        """Adds the layer as first layer.

        Parameters
        ----------
        index : int
            It is the index at which the layer must be inserted. Layers from
            index `index` will be moved forward. The index can also be given as
            negative index as in standard python lists.

        layer_spec : Tuple[str, BaseModel, bool] | Tuple[str, BaseModel] | Tuple[BaseModel, bool] | BaseModel
            A layer specified as it is specified in the init of the pipeline as
            a list element.

        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If index is not int.
        
        IndexError
            If index is out of bounds.
        
        ValueError
            If the tuple length is greater than 3 or if the object passed is
            neither tuple nor BaseModel.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def append_layer(self, layer_spec: Tuple[str, BaseModel, bool] | Tuple[str, BaseModel] | Tuple[BaseModel, bool] | BaseModel) -> None:
        """Adds the layer as last layer.

        Parameters
        ----------
        layer_spec : Tuple[str, BaseModel, bool] | Tuple[str, BaseModel] | Tuple[BaseModel, bool] | BaseModel
            A layer specified as it is specified in the init of the pipeline as
            a list element.

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the tuple length is greater than 3 or if the object passed is
            neither tuple nor BaseModel.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def remove_layer(self, layer: int | str, recursive: bool = True) -> None:
        """Remove the specified layer.

        Parameters
        ----------
        layer : int or str
            It is the position in the pipeline or the name of the layer to
            remove.

        recursive : bool, default=True
            It states whether the linked objects to the deleted layer must be
            deleted too. Let's say we have a layer `pre_a` and a layer `post_b`.
            If the layer `post_b` has a member with a reference to `pre_a` and
            we delete `pre_a` with `recursive=True`, also `post_b` will be
            deleted. This notion generalizes to an object that is referenced by
            multiple layers in the pipeline.

        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If layer is neither int nor str or if name is not str.
            
        IndexError
            If the integer or string passed to identify the layer is out of
            bounds or there are no layers with that name.

        Notes
        -----
        Circular references are not a problem, first the pipeline is inspected
        to find all layers that must be deleted saving their names or position
        in the pipeline. Then, once all the layers have been identified, they
        will be deleted.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def process(self, x, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Process all the pipeline objects and outputs the result.
        
        Parameters
        ----------
        x : array-like
            The input objects to the pipeline.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        processed_values : ndarray
            The processed values for the input `x`. The input is passed through
            each layer of the pipeline and the output of the last layer is
            returned.
            
        processed_targets : ndarray
            The created values for the target from the input. It might be None
            if there are no shape change layers or more in general layers
            producing both `x` and `y` from a single `x`. This may be useful to
            create a pipeline with many shape changer.
            
        Raises
        ------
        InvalidInputShape
            If the input to a layer has wrong shape.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def summary(self) -> None:
        """Prints a summary of the pipeline.
        
        It prints all the layers with the respective names in order from top to
        bottom.
        
        Returns
        -------
        None
        """
        raise NotImplementedError
