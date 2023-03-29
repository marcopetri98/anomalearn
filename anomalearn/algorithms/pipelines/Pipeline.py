from __future__ import  annotations

from pathlib import Path
from typing import Tuple, Any

import numpy as np

from . import IPipeline, IPipelineLayer
from .. import IPredictor, SavableModel, IShapeChanger, ITransformer, BaseModel, \
    ICluster, IClassifier, IRegressor, IParametric, load_estimator, \
    instantiate_estimator
from ...exceptions import InvalidInputShape
from ...input_validation import is_var_of_type
from ...utils import save_py_json, load_py_json


class Pipeline(IPipeline):
    """The basic pipeline object.
    
    The pipeline object takes a sequence of transformers, shape changer,
    predictors and parametric objects. The input is passed through each of them
    to produce the output. The objects have a name to easy the access to the
    object's parameter and hyperparameters. A transformer is an object which
    works on values of `x`. A shape changer is an object which may change the
    shape of `x`, of `y` or both. A parametric object is an object that can
    learn at least one parameter (such as some normalization objects). A
    predictor is an object which can predict values given some input. Every
    object to be considered like that must implement the appropriate interfaces.
    
    Parameters
    ----------
    elements : list[Tuple[str, IPipelineLayer, bool] | Tuple[str, IPipelineLayer] | Tuple[IPipelineLayer, bool] | IPipelineLayer], default=None
        It is the list of elements composing the pipeline. The pipeline can be
        composed of `ITransformer`, `IShapeChanger`, `IParametric`, `ICluster`,
        `IClassifier`, `IRegressor` or `IPredictor` objects, provided that they
        are `IPipelineLayer`. The input to the pipeline is passed to each of these in
        the order they are inserted in the pipeline. Transformers do not change
        the shape and only work on `x`. Shape changers receives in input both `x`
        and `y` and outputs `new_x` and `new_y`. Parametric objects are those
        which must be fitted. They receive the last processed `x` and `y` inputs
        and fit to them. Predictors are objects which predict a value given an
        input. Objects which are both parametric and another object are treated
        like that:
        
        * If the fit has been called on the pipeline, first the fit on the
          object is called, then its function.
        * If the `predict` has been called on the pipeline, only their specific
          function is called (such as predict, transform or shape_change).
          
        The first element of the tuple is the name of the "layer" and can be
        omitted. If the name is omitted, the layer will be called with class
        name and an increasing number equal to its index in the list. The last
        element of the tuple is a boolean stating if the "layer" must be fit or
        not and can be omitted. If the train boolean is omitted, it is assumed
        to be True. If it is None, an empty Pipeline is instantiated.
        
    Attributes
    ----------
    _elements : list[Tuple[str, IPipelineLayer, bool]]
        The final and used list of layers used by the pipeline internally.
        
    _layer_num : int
        The number of layers that have been added to the pipeline. It is used
        to keep the names of the layers unique and different.
        
    Notes
    -----
    The objects in the pipeline whose type is `ICluster`, `IClassifier`,
    `IRegressor` or `IPredictor` must not overlap. Therefore, if an object is
    a `ICluster`, it can't be `IClassifier`, `IRegressor` or `IPredictor`. The
    pipeline will still work with undefined behaviour highly subject to changes.
    In such a case the pipeline try to get from the layer the interface to be
    used. If a valid interface is returned, it will be used. Otherwise,
    undefined behaviour is performed.
    
    The objects in the pipeline whose type is `ITransformer` or `IShapeChanger`
    must not overlap. Therefore, it must be the former or the latter. The
    pipeline will still work undefined behaviour highly subject to changes.
    
    All objects implementing the `IParametric` interface will be fitted first
    and processed secondly during pipeline fit. During pipeline process only the
    processing operation will be called.
    """
    __json_file = "pipeline.json"
    __json_references = "pipeline_layers_references.json"
    __to_load_str = "to_load"
    __to_create_str = "to_create"
    
    def __init__(self, elements: list[Tuple[str, IPipelineLayer, bool] | Tuple[str, IPipelineLayer] | Tuple[IPipelineLayer, bool] | IPipelineLayer] = None):
        super().__init__()
        
        if elements is None:
            elements = []
        
        # manage optional strings and bools
        self._layer_num = 0
        self._elements = []
        for e in elements:
            self.append_layer(e)
            
        if len(self.pipeline_names) != len(set(self.pipeline_names)):
            raise ValueError("there are layers with the same name. Each layer "
                             "must have a unique name.")
        
        # check that the list of elements is correct
        for name, obj, train in self._elements:
            if not isinstance(name, str):
                raise TypeError("the first element of the tuples must be a str")
            elif not is_var_of_type(obj, [ITransformer, IShapeChanger, IParametric, IPredictor]):
                raise TypeError("the second element of the tuples must be one "
                                "of ITransformer, IShapeChanger, IParametric or"
                                " IPredictor")
            elif not isinstance(train, bool):
                raise TypeError("the last (third) element of the tuples must be"
                                " a bool")
        
    @property
    def pipeline_spec(self):
        return self._elements.copy()
    
    @property
    def pipeline_names(self):
        return [e[0] for e in self._elements]
    
    @property
    def pipeline_layers(self):
        return [e[1] for e in self._elements]
    
    @property
    def pipeline_train(self):
        return [e[2] for e in self._elements]
    
    def allowed_interfaces(self) -> list:
        return [ITransformer, IShapeChanger, ICluster, IClassifier, IRegressor, IPredictor, IParametric]
    
    def set_name(self, layer: int | str, name: str) -> None:
        if not isinstance(layer, int) and not isinstance(layer, str):
            raise TypeError("layer must be an int or a str")
        elif not isinstance(name, str):
            raise TypeError("name must be a string")
        elif isinstance(layer, str) and layer not in self.pipeline_names:
            raise IndexError(f"if str layer must be one of {self.pipeline_names} for this pipeline")
        elif name != layer and name in self.pipeline_names:
            raise ValueError("name must not be a name of another layer")
        
        if isinstance(layer, str):
            pos = self.pipeline_names.index(layer)
        else:
            pos = layer
            
        self._elements[pos] = (name, self._elements[pos][1], self._elements[pos][2])
    
    def set_trainable(self, layer: int | str, trainable: bool) -> None:
        if not isinstance(layer, int) and not isinstance(layer, str):
            raise TypeError("layer must be an int or a str")
        elif not isinstance(trainable, bool):
            raise TypeError("name must be a bool")
        elif isinstance(layer, str) and layer not in self.pipeline_names:
            raise IndexError(f"if str layer must be one of {self.pipeline_names} for this pipeline")
        
        if isinstance(layer, str):
            pos = self.pipeline_names.index(layer)
        else:
            pos = layer
            
        self._elements[pos] = (self._elements[pos][0], self._elements[pos][1], trainable)

    def add_first_layer(self, layer_spec: Tuple[str, IPipelineLayer, bool] | Tuple[str, IPipelineLayer] | Tuple[IPipelineLayer, bool] | IPipelineLayer) -> None:
        self.insert_layer(0, layer_spec)

    def insert_layer(self, index: int,
                     layer_spec: Tuple[str, IPipelineLayer, bool] | Tuple[str, IPipelineLayer] | Tuple[IPipelineLayer, bool] | IPipelineLayer) -> None:
        if not isinstance(index, int):
            raise IndexError("The index must be an integer.")
    
        final_model_tuple = self._get_final_model_tuple(layer_spec)
        
        if final_model_tuple[0] in self.pipeline_names:
            raise ValueError("layers' names must be unique")
        
        try:
            self._elements.insert(index, final_model_tuple)
            self._layer_num += 1
        except IndexError as e:
            raise IndexError(f"The index is out of range. Pipeline has len={len(self)}")

    def append_layer(self, layer_spec: Tuple[str, IPipelineLayer, bool] | Tuple[str, IPipelineLayer] | Tuple[IPipelineLayer, bool] | IPipelineLayer) -> None:
        self.insert_layer(len(self._elements), layer_spec)

    def remove_layer(self, layer: int | str, recursive: bool = True) -> None:
        if not isinstance(layer, int) and not isinstance(layer, str):
            raise TypeError("The index must be an integer or a string.")
        elif isinstance(layer, int) and not 0 < layer < len(self):
            raise IndexError(f"The index is out of range. Pipeline has len={len(self)}")
        elif isinstance(layer, str) and layer not in self.pipeline_names:
            raise ValueError(f"The pipeline does not have a layer called {layer}")
    
        if isinstance(layer, str):
            layer_name = layer
            layer_pos = self.pipeline_names.index(layer)
        else:
            layer_pos = layer
            layer_name = self.pipeline_names[layer]
        layer_obj = self._elements[layer_pos][1]
        
        if recursive:
            layers_to_delete = self._get_connected_layers(layer_name, layer_obj)
            layers_to_delete_pos = list(reversed(sorted([self.pipeline_names.index(e) for e in layers_to_delete])))
        else:
            layers_to_delete_pos = [layer_pos]
        
        for layer_pos in layers_to_delete_pos:
            del self._elements[layer_pos]
        
    def _get_connected_layers(self, referenced: str,
                              referenced_obj: IPipelineLayer,
                              found_layers: list[str] = None) -> list[str]:
        """Found all the layers connected to `referenced` in the pipeline.
        
        Parameters
        ----------
        referenced : str
            It is the name of the layer for which we want to find all the
            objects pointing to it.
            
        referenced_obj : IPipelineLayer
            It is the layer object with name `referenced`.
        
        found_layers : list[str], default=None
            It is the list of all the layers currently found. It is used to
            avoid ending up in infinite call chain due to cycles in dependencies.
            If layer `a` points to `b` and `b` points to `a`, we have a circular
            dependency. We want to save each layer only once.

        Returns
        -------
        connected_layers : list[str]
            All the layers connected to `referenced` and `referenced`. Thus, it
            is a list with minimum length of 1.
        """
        if found_layers is None:
            found_layers = [referenced]
            
        referencing_layers = list(self._get_referencing_layers(referenced, referenced_obj).keys())
        
        for name, obj, _ in self._elements:
            if name in referencing_layers and name not in found_layers:
                found_layers.append(name)
                found_layers = self._get_connected_layers(name, obj, found_layers)
        
        return found_layers
        
    def _get_referencing_layers(self, referenced: str,
                                referenced_obj: IPipelineLayer) -> dict:
        """Gets all the layers of the pipeline referencing the `referenced` layer.
        
        Parameters
        ----------
        referenced : str
            The name of the layer that is "pointed" from a reference inside a
            member of another pipeline's layer.
        
        referenced_obj : IPipelineLayer
            The object that is "pointed".

        Returns
        -------
        referencing_members : dict[str: list[str]]
            It is a dictionary containing all the members of the layers pointing
            to `referenced`. The key is the name of the layer pointing to
            `referenced` and the value is the list of members' names with a
            reference to `referenced`.
        """
        referencing_members = dict()
        
        for name, obj, _ in self._elements:
            if name != referenced:
                members = vars(obj)
                for m_name, m_value in members.items():
                    if m_value is referenced_obj:
                        if name not in referencing_members.keys():
                            referencing_members[name] = [m_name]
                        else:
                            referencing_members[name].append(m_name)
                        
        return referencing_members
        
    def __len__(self):
        return len(self._elements)
    
    def __repr__(self):
        representation = "Pipeline(["
        
        for i, e in enumerate(self._elements):
            representation += f"({e[0]}, {repr(e[1])}, {e[2]})"
            
            if i != len(self._elements) - 1:
                representation += ",\n          "
                
        representation += "])"
                
        return representation
    
    def summary(self) -> None:
        names, objs, train = list(zip(*self._elements))
        names = list(names)
        objs_strings = [str(e) for e in objs]
        input_shapes = [str(e.get_input_shape()) for e in objs]
        output_shapes = [str(e.get_output_shape()) for e in objs]
        train_strings = ["True " if e else "False" for e in train]
        
        max_len_names = max([len(e) for e in names] + [len("Layer name")])
        max_len_objs = max([len(e) for e in objs_strings] + [len("Layer")])
        max_len_in_shapes = max([len(e) for e in input_shapes] + [len("Input shape")])
        max_len_out_shapes = max([len(e) for e in output_shapes] + [len("Output shape")])
        
        width = 2 + max_len_names + 3 + max_len_objs + 3 + max_len_in_shapes + 3 + max_len_out_shapes + 3 + 5 + 2
        
        left = (width - 4 - len("Pipeline summary")) // 2
        right = left if left * 2 == width - 4 - len("Pipeline summary") else left + 1
        summary = "-" * width + "\n"
        summary += "|" + " " * (width - 2) + "|\n"
        summary += "| " + " " * left + "Pipeline summary" + " " * right + " |\n"
        summary += "|" + " " * (width - 2) + "|\n"
        summary += "-" * width + "\n"
        for name, obj, in_shape, out_shape, train in zip(["Layer name"] + names,
                                                         ["Layer"] + objs_strings,
                                                         ["Input shape"] + input_shapes,
                                                         ["Output shape"] + output_shapes,
                                                         ["Train"] + train_strings):
            left = (max_len_names - len(name)) // 2
            right = left if left * 2 == max_len_names - len(name) else left + 1
            summary += "| " + " " * left + name + " " * right

            left = (max_len_objs - len(obj)) // 2
            right = left if left * 2 == max_len_objs - len(obj) else left + 1
            summary += " | " + " " * left + obj + " " * right

            left = (max_len_in_shapes - len(in_shape)) // 2
            right = left if left * 2 == max_len_in_shapes - len(in_shape) else left + 1
            summary += " | " + " " * left + in_shape + " " * right

            left = (max_len_out_shapes - len(out_shape)) // 2
            right = left if left * 2 == max_len_out_shapes - len(out_shape) else left + 1
            summary += " | " + " " * left + out_shape + " " * right
            
            summary += " | " + train + " |\n"
            summary += "-" * width + "\n"
        
        print(summary)
    
    def __str__(self):
        return "Pipeline"
    
    def __eq__(self, other):
        if not isinstance(other, Pipeline):
            return False
        elif len(self) != len(other):
            return False
        
        if self.pipeline_layers != other.pipeline_layers:
            return False
            
        for (name1, obj1), (name2, obj2) in zip(zip(self.pipeline_names, self.pipeline_layers), zip(other.pipeline_names, other.pipeline_layers)):
            this_ref_dict = self._get_referencing_layers(name1, obj1)
            other_ref_dict = other._get_referencing_layers(name2, obj2)
            this_ref_names = list(this_ref_dict.keys())
            other_ref_names = list(other_ref_dict.keys())
            
            if len(this_ref_names) != len(other_ref_names):
                return False
            
            for this_ref, other_ref in zip(this_ref_names, other_ref_names):
                if self.pipeline_names.index(this_ref) != other.pipeline_names.index(other_ref):
                    return False
                
            for (_, this_members), (_, other_members) in zip(this_ref_dict.items(), other_ref_dict.items()):
                if this_members != other_members:
                    return False
        
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def identical(self, other, degree: int = 2) -> bool:
        if degree != 1 and degree != 2 and degree != 3:
            raise ValueError("degree must be either 1, 2 or 3")
        
        if degree == 1:
            return self.__eq__(other)
        else:
            if not self.__eq__(other):
                return False
        
            return self.pipeline_train == other.pipeline_train and (degree != 3 or self.pipeline_names == other.pipeline_names)
        
    def copy(self) -> Pipeline:
        elements_copy = []
        for name, obj, train in self._elements:
            elements_copy.append((name, obj.copy(), train))
        new = Pipeline(elements_copy)
        
        for name, obj, _ in self._elements:
            pointed = new.pipeline_layers[self.pipeline_names.index(name)]
            refs_dict = self._get_referencing_layers(name, obj)
            
            for key, value in refs_dict.items():
                pointer = new.pipeline_layers[self.pipeline_names.index(key)]
                for member in value:
                    setattr(pointer, member, pointed)
        
        return new
            
    def save(self, path,
             *args,
             **kwargs) -> Any:
        super().save(path=path)
        path_obj = Path(path)
        
        pipeline_structure = []
        layers_references = dict()
        for name, obj, train in self._elements:
            if isinstance(obj, SavableModel):
                pipeline_structure.append((name, obj.__class__.__name__, self.__to_load_str, train))
            else:
                pipeline_structure.append((name, obj.__class__.__name__, self.__to_create_str, train))
                
            ref_dict = self._get_referencing_layers(name, obj)
            layers_references[name] = ref_dict
            
        save_py_json(pipeline_structure, str(path_obj / self.__json_file))
        save_py_json(layers_references, str(path_obj / self.__json_references))
        
        for layer_name, _, method, _ in pipeline_structure:
            if method == self.__to_load_str:
                obj = self.pipeline_layers[self.pipeline_names.index(layer_name)]
                obj.save(str(path_obj / layer_name))
    
    def load(self, path: str,
             estimator_classes: list = None,
             *args,
             **kwargs) -> Any:
        """Loads a serialized pipeline.
        
        Parameters
        ----------
        path : str
            It is the path of the directory in which the object has been saved.
            
        estimator_classes : list, default=None
            It is the list of external classes not present in the library that
            may be loaded or instantiated by the pipeline at some point.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        self
            Instance to itself to allow chain calls.
    
        Raises
        ------
        ValueError
            If the given path does not point to a saved model.
        """
        super().load(path=path)
        path_obj = Path(path)
        
        pipeline_structure = load_py_json(str(path_obj / self.__json_file))
        layers_references = load_py_json(str(path_obj / self.__json_references))
        
        # rebuild the pipeline with layers' names and values
        self._elements = []
        for layer_name, obj_class, method, train in pipeline_structure:
            if method == self.__to_load_str:
                obj = load_estimator(str(path_obj / layer_name), estimator_classes=estimator_classes)
            else:
                obj = instantiate_estimator(obj_class, estimator_classes=estimator_classes)
            self.append_layer((layer_name, obj, train))
        
        # rebuild pipeline's layers' references
        for referenced, refs in layers_references.items():
            obj = self.pipeline_layers[self.pipeline_names.index(referenced)]
            
            for layer_name, member in refs.items():
                pointer = self.pipeline_layers[self.pipeline_names.index(layer_name)]
                for m_name in member:
                    setattr(pointer, m_name, obj)
                
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> Pipeline:
        obj = Pipeline([])
        obj.load(path)
        return obj
    
    def get_input_shape(self) -> tuple:
        if len(self._elements) != 0:
            return self.pipeline_layers[0].get_input_shape()
        else:
            return tuple()
    
    def get_output_shape(self) -> tuple:
        if len(self._elements) != 0:
            return self.pipeline_layers[-1].get_output_shape()
        else:
            return tuple()
        
    def get_hyperparameters(self, *args, **kwargs) -> dict:
        """Gets the hyperparameters present in the pipeline.
        
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
            layer of the pipeline and the value is the result of the method
            `get_hyperparameters` on the layer.
        """
        hyperparameters = dict()
        for e in self._elements:
            hyperparameters[e[0]] = e[1].get_hyperparameters()
        return hyperparameters
    
    def set_hyperparameters(self, hyperparameters: dict, *args, **kwargs) -> None:
        """Sets the hyperparameters in the pipeline.
        
        Parameters
        ----------
        hyperparameters : dict
            The keys are the names of the layers in the pipeline and the value
            is the dictionary of the hyperparameters to set for the layer in the
            format accepted by `set_hyperparameter`.
        
        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        for name, obj, train in self._elements:
            if name in hyperparameters.keys():
                obj.set_hyperparameters(hyperparameters[name])
        
    def fit(self, x,
            y=None,
            checkpoint_folder=None,
            *args,
            **kwargs) -> None:
        """Fits the pipeline by fitting the layers that can be fit.
        
        The pipeline iterates over the layers from the first till the last. If a
        layer is parametric and its train flag is True, it is fitted with the
        data produced up to that point. Then, its processing function will be
        called. Otherwise, only the processing function will be called. After
        each training
        
        Parameters
        ----------
        checkpoint_folder : path-like, default=None
            It is the path in which the pipeline must store the checkpoints of
            training process.

        Returns
        -------
        None
        """
        x = np.array(x)
        y = np.array(y) if y is not None else None
        
        curr_x, curr_y = (x.copy(), y.copy()) if y is not None else (x.copy(), None)
        for i, (name, obj, train) in enumerate(self._elements):
            if isinstance(obj, IParametric) and train:
                obj.fit(curr_x, curr_y)
                # after the layer has been fit, set its flag to avoid training
                # therefore if the model pipeline gets an error at some point,
                # the model can be saved with a partial fit and reloaded once
                # the pipeline is fixed and resume from the same point
                self._elements[i] = (self._elements[i][0], self._elements[i][1], False)
                
                if checkpoint_folder is not None:
                    check_folder = Path(checkpoint_folder) / ("_" + str(i))
                    self.save(check_folder)
            
            curr_x, curr_y = self._execute_layer(name, obj, curr_x, curr_y)
    
    def process(self, x,
                *args,
                **kwargs) -> tuple[np.ndarray, np.ndarray]:
        x = np.array(x)
        if "y" in kwargs:
            y = np.array(kwargs["y"])
        else:
            y = None
        
        curr_x = x
        curr_y = y
        for name, obj, train in self._elements:
            curr_x, curr_y = self._execute_layer(name, obj, curr_x, curr_y)
            
        return curr_x, curr_y
    
    def _get_final_model_tuple(self, model_tuple: Tuple[str, IPipelineLayer, bool] | Tuple[str, IPipelineLayer] | Tuple[IPipelineLayer, bool] | IPipelineLayer) -> Tuple[str, IPipelineLayer, bool]:
        """Gets the correct final model tuple.
        
        Parameters
        ----------
        model_tuple : Tuple[str, IPipelineLayer, bool] | Tuple[str, IPipelineLayer] | Tuple[IPipelineLayer, bool] | IPipelineLayer
            The model tuple to complete.

        Returns
        -------
        final_model_tuple : Tuple[str, IPipelineLayer, bool]
            The complete and correct model tuple with all the fields.
        """
        new_tuple = None
        
        try:
            if isinstance(model_tuple, BaseModel) or len(model_tuple) == 1:
                if isinstance(model_tuple, BaseModel):
                    model = model_tuple
                else:
                    model = model_tuple[0]
        
                new_tuple = (str(model) + "_" + str(self._layer_num), model, True)
            elif len(model_tuple) > 3:
                raise ValueError("a model tuple must be at most of length 3")
            elif len(model_tuple) == 2:
                if isinstance(model_tuple[1], BaseModel):
                    new_tuple = (model_tuple[0], model_tuple[1], True)
                else:
                    new_tuple = (str(model_tuple[0]) + "_" + str(self._layer_num), model_tuple[0], model_tuple[1])
            else:
                new_tuple = model_tuple
        except TypeError:
            new_tuple = (str(model_tuple) + "_" + str(self._layer_num), model_tuple, True)
            
        return new_tuple
    
    def _execute_layer(self, layer_name: str,
                       layer: IPipelineLayer,
                       x: np.ndarray,
                       y: np.ndarray | None,
                       *args,
                       **kwargs) -> Tuple[np.ndarray, np.ndarray | None]:
        """Execute a layer.
        
        Parameters
        ----------
        layer_name : str
            The name of the layer to execute.
            
        layer : IPipelineLayer
            The layer object to execute.
        
        x : ndarray
            The current `x` to be used.
        
        y : ndarray
            The current `y` to be used.
        
        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        new_x : ndarray
            It is the result of the processing operations of the layer to the
            input `x`.
        
        new_y : ndarray or None
            It is the result of the processing operations of the layer to the
            input `y` or the input `y` itself if the layer does not change `y`.
        """
        new_x, new_y = x, y
        
        try:
            if isinstance(layer, IPipeline):
                new_x, new_y = layer.process(new_x, y=new_y)
            else:
                num_of_preprocess = 0
                for interface in [ITransformer, IShapeChanger]:
                    if isinstance(layer, interface):
                        num_of_preprocess += 1
                
                klass = None
                if num_of_preprocess > 1:
                    klass = layer.get_pipeline_class()
                
                    if klass is not None:
                        if klass is ITransformer:
                            new_x = layer.transform(new_x)
                        elif klass is IShapeChanger:
                            new_x, new_y = layer.shape_change(new_x, new_y)
                    
                if klass is None:
                    if isinstance(layer, ITransformer):
                        new_x = layer.transform(new_x)
                    elif isinstance(layer, IShapeChanger):
                        new_x, new_y = layer.shape_change(new_x, new_y)
                        
                num_of_models = 0
                for interface in [ICluster, IClassifier, IRegressor, IPredictor]:
                    if isinstance(layer, interface):
                        num_of_models += 1
            
                klass = None
                if num_of_models > 1:
                    klass = layer.get_pipeline_class()
                    
                    if klass is not None:
                        if klass is ICluster:
                            new_x = layer.cluster(new_x)
                        elif klass is IClassifier:
                            new_x = layer.classify(new_x)
                        elif klass is IRegressor:
                            new_x = layer.regress(new_x)
                        elif klass is IPredictor:
                            new_x = layer.predict(new_x)
            
                if klass is None:
                    if isinstance(layer, ICluster):
                        new_x = layer.cluster(new_x)
                    elif isinstance(layer, IClassifier):
                        new_x = layer.classify(new_x)
                    elif isinstance(layer, IRegressor):
                        new_x = layer.regress(new_x)
                    elif isinstance(layer, IPredictor):
                        new_x = layer.predict(new_x)
        except InvalidInputShape as e:
            new_msg = f"The input of layer {layer_name} received wrong input shape." + e.message
            raise InvalidInputShape(e.expected_shape, e.shape, new_msg)
            
        return new_x, new_y
