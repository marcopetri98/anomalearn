from .. import SavableModel
from . import IPipelineLayer


class AbstractPipelineSavableLayer(IPipelineLayer, SavableModel):
    """A pipeline layer that is savable.
    """
