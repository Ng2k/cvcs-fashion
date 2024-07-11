"""
@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from typing import List, Tuple

class MaskUtility():
    """Classe per le utility del modulo della generazione delle maschere
    """
    _LABEL_BLACK_LIST: List[int] = [0, 2, 11, 12, 13, 14, 15]
    RIGHT_SHOE_INDEX = 10
    LEFT_SHOE_INDEX = 9

    @staticmethod
    def get_label_black_list() -> List[int]:
        """Ritorna la lista di label da ignorare per la creazione di maschere

        Returns:
        -------
            List[int]: lista di label
        """
        return MaskUtility._LABEL_BLACK_LIST

    @staticmethod
    def get_shoes_index() -> Tuple[int, int]:
        """Ritorna gli indici delle scarpe

        Returns:
        -------
            Tuple[int, int]: (right shoe, left shoe)
        """
        return (MaskUtility.RIGHT_SHOE_INDEX, MaskUtility.LEFT_SHOE_INDEX)
