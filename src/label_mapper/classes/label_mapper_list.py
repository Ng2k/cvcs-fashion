"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from typing import List
from functools import partial

from label_mapper.interfaces.label_mapper import ILabelMapper
from mask_generator.types.list_mask_type import ListMaskType

class LabelMapperList(ILabelMapper):
    """Classe per il mapping delle label custom alle maschere in formato List
    """

    def map_labels_to_masks(self, masks: ListMaskType, prompt_list: List[str]) -> List[dict]:
        return list(map(
            partial(self._mapper, prompt_list=prompt_list),
            list(map(lambda x: x['mask'], masks))
        ))
