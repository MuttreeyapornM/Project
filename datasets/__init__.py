# from .voc import VOCSegmentation
# from .cityscapes import Cityscapes

from .voc import VOCSegmentation
from .cityscapes import Cityscapes
from .custom_dataset import CustomSegmentation

__all__ = ['VOCSegmentation', 'Cityscapes', 'CustomSegmentation']