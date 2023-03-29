from . import IPipelineLayer
from .. import BaseModel


class AbstractPipelineBaseLayer(IPipelineLayer, BaseModel):
    """A pipeline layer that isn't savable.
    """
