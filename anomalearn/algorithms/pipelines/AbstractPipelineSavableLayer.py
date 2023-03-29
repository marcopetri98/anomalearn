from . import IPipelineLayer
from .. import SavableModel


class AbstractPipelineSavableLayer(IPipelineLayer, SavableModel):
    """A pipeline layer that is savable.
    """
