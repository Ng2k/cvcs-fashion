"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import torch

class Utils():
    """Classe con funzioni di utiliti generali
    """
    @staticmethod
    def get_device() -> str:
        """
        Restituisce il device su cui eseguire il modello.

        Returns:
        ----------
            str
                Il device su cui eseguire il modello.
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'