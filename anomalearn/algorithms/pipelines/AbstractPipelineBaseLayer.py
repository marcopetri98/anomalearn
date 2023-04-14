from .. import BaseModel
from . import IPipelineLayer


class AbstractPipelineBaseLayer(IPipelineLayer, BaseModel):
    """A pipeline layer that isn't savable.
    """
