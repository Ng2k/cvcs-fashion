"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from functools import partial
from typing import List

from src.label_mapper.interfaces.label_mapper import ILabelMapper
from src.mask_generator.types.dict_mask_type import DictMaskType

class LabelMapperDict(ILabelMapper):
    """Classe per il mapping delle label custom alle maschere in formato List
    """

    def map_labels_to_masks(self, masks: DictMaskType, prompt_list: List[str]) -> List[dict]:
        return list(map(
            partial(self._mapper, prompt_list=prompt_list),
            masks.values()
        ))
