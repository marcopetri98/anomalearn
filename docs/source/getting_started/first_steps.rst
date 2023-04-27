.. _getting_started:

===============
Getting started
===============

Installation
============

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :text-align: center
      :class-header: card-title

      Do you use pip?
      ^^^

      anomalearn can be installed via pip since it is published on
      `PyPI <https://pypi.org/project/anomalearn/>`_

      +++
      .. code-block:: bash

         pip install anomalearn

   .. grid-item-card::
      :text-align: center
      :class-header: card-title

      Do you use conda?
      ^^^

      Currently, anomalearn has not been published on conda, but we plan to do
      so in the future.

      +++
      Not yet available

.. grid:: 1
   :gutter: 3

   .. grid-item-card::
      :text-align: left
      :class-header: card-small-title

      Not satisfied with previous instructions?
      ^^^

      Install from source? Do you want to know dependencies? Check the detailed
      installation tutorial.

      .. button-ref:: advanced_install
         :ref-type: ref
         :color: primary
         :align: left
         :click-parent:

         Go to installation tutorial

Quick facts
===========

.. dropdown:: What is the purpose of anomalearn?
   :animate: fade-in-slide-down

   The purpose of anomalearn is to ship a complete structure to ease the
   development of new methods for anomaly detection. It wants to aid the
   process of creating new models, pre-processing operations, post-processing
   operations, transformers, pipelines or hyperparameter search algorithms.

   :bdg-ref-primary:`To introduction tutorial <intro_purpose>`

.. dropdown:: What kind of algorithms does anomalearn contain?
   :animate: fade-in-slide-down

   anomalearn will contain several types of algorithms that are directly used to
   perform anomaly detection (such as isolation forest), or algorithms which
   does not solve anomaly detection, but that are surrounded by processing
   objects or other algorithms to carry on the anomaly detection task.

   :bdg-ref-primary:`To introduction tutorial <intro_algorithms>`

.. dropdown:: What kind of experiment helpers does anomalearn contain?
   :animate: fade-in-slide-down

   anomalearn will contain objects to aid the creation of experiments to aid
   scientific research in computer science. An example of helper for experiment
   creation is the loader of datasets, which enables to easily iterate over
   training/testing sets, either the default (when the dataset contain a
   pre-defined split) or a specifically user defined split.

   :bdg-ref-primary:`To introduction tutorial <intro_experiments>`

.. dropdown:: How do I read public datasets with anomalearn?
   :animate: fade-in-slide-down

   One of the features offered by anomalearn is that of dataset readers. These
   components enable the user to instantiate a data reader object to read the
   public dataset. These objects can be iterated and indexed with usual python
   syntax. The return type is always a
   :external+pandas:py:class:`pandas.DataFrame`.

   :bdg-ref-primary:`To introduction tutorial <intro_readers>`

.. dropdown:: How do I analyse data with anomalearn?
   :animate: fade-in-slide-down

   The analysis functions offered by anomalearn are all functions present in the
   analysis package. These functions needs data in input and have some default
   parameters to specify how the analysis must be performed. Some of the
   analysis functions are implemented using :external+numba:py:mod:`numba`
   to be efficient.

   :bdg-ref-primary:`To introduction tutorial <intro_analysis>`

.. dropdown:: How do I perform hyperparameter search with anomalearn?
   :animate: fade-in-slide-down

   anomalearn contains several algorithms to do research on time series anomaly
   detection. The :py:mod:`anomalearn.algorithms.tuning.hyperparameter` package
   contains all the components related to hyperparameter tuning. Searchers
   can be instantiated to search for optimal parameters of any function.

   :bdg-ref-primary:`To introduction tutorial <intro_hyperparameter>`

.. dropdown:: How do I integrate my legacy objects with anomalearn?
   :animate: fade-in-slide-down

   anomalearn follows a type-based approach with interfaces defining the
   methods. If your objects implement functions such as
   :py:meth:`~anomalearn.algorithms.IPredictor.IPredictor.predict`, you can
   simply inherit from the function and it will be integrated. In some cases,
   interfaces also enable duck typing, which let any legacy object to work
   with anomalearn without the actual need of inheriting anomalearn's objects.

   :bdg-ref-primary:`To introduction tutorial <intro_legacy>`

.. dropdown:: How do I create a model with anomalearn?
   :animate: fade-in-slide-down

   anomalearn models inherit from at least one of the classes (concrete or
   abstract) in :py:mod:`anomalearn.algorithms`. Each model in anomalearn must
   inherit from :py:class:`~anomalearn.algorithms.BaseModel.BaseModel`, or from
   :py:class:`~anomalearn.algorithms.SavableModel.SavableModel` if it can be
   safely serialized to file.

   :bdg-ref-primary:`To introduction tutorial <intro_models>`

.. dropdown:: How do I create a pre-processing object with anomalearn?
   :animate: fade-in-slide-down

   anomalearn pre-processing objects inherit from interfaces located in
   :py:mod:`anomalearn.algorithms`. These models typically inherit from the
   :py:class:`~anomalearn.algorithms.IShapeChanger.IShapeChanger` interface, and
   should not be confused with transformers. They implement operations which
   happen always before a model and make sense only if coupled with a model.

   :bdg-ref-primary:`To introduction tutorial <intro_preprocessing>`

.. dropdown:: How do I create a post-processing object with anomalearn?
   :animate: fade-in-slide-down

   anomalearn post-processing objects inherit from interfaces located in
   :py:mod:`anomalearn.algorithms`. These models typically inherit from the
   :py:class:`~anomalearn.algorithms.IShapeChanger.IShapeChanger` interface, and
   should not be confused with transformers. They implement operations which
   happen always after a model and make sense only if coupled with a model.
   Sometimes they also depend on the pre-processing placed before the model.

   :bdg-ref-primary:`To introduction tutorial <intro_postprocessing>`

.. dropdown:: How do I create a transformer with anomalearn?
   :animate: fade-in-slide-down

   anomalearn transformers are objects which transform data without changing
   the shape of data, e.g.
   :py:class:`anomalearn.algorithms.transformers.MinMaxScaler`. These objects
   inherit from :py:class:`~anomalearn.algorithms.ITransformer.ITransformer`.


   :bdg-ref-primary:`To introduction tutorial <intro_transformers>`

.. dropdown:: What is the difference between pre-, post-processing and transformers?
   :animate: fade-in-slide-down

   The first difference between processing and transformer objects is that the
   latter does not change the shape of the data in input. The former type of
   object explicitly changes the shape of the objects in input. However, both
   processing and transformer objects manipulate data. The difference between
   pre- and post- processing objects is that the former always happens before
   the model, the latter always happens after the model.

   :bdg-ref-primary:`To introduction tutorial <intro_difference_processing>`

.. dropdown:: How do I create a pipeline with anomalearn?
   :animate: fade-in-slide-down

   anomalearn contains the implementation of a sequential pipeline which can
   contain any sequence of objects following the layer interface. The
   :py:class:`~anomalearn.algorithms.pipelines.Pipeline.Pipeline` object can
   be created with a list of layers, and it is mutable. Any sequence of objects
   whose input/output shape are coherent can be created, and pipelines are
   layers themselves, i.e. a pipeline can contain another pipeline.

   :bdg-ref-primary:`To introduction tutorial <intro_pipeline>`

.. dropdown:: Has anomalearn any helper function?
   :animate: fade-in-slide-down

   Yes, anomalearn contains several helper functions for the purpose of
   development of its main features. The helper packages are:

   - :py:mod:`anomalearn.utils`: contains functions to work with other objects,
     such as :external+numpy:py:class:`numpy.ndarray` or
     :external+pandas:py:class:`pandas.DataFrame`.
   - :py:mod:`anomalearn.input_validation`: contains functions to validate the
     input to functions and methods.
   - :py:mod:`anomalearn.visualizer`: contains functions to plot time series
     and results of detections.
   - :py:mod:`anomalearn.exceptions`: contains detailed exceptions with
     configurable messages to give detailed information regarding the error
     that occurred.

   :bdg-ref-primary:`To introduction tutorial <intro_helpers>`

.. toctree::
   :maxdepth: 3
   :hidden:

   advanced_install
   intro/tutorials
   package_overview
