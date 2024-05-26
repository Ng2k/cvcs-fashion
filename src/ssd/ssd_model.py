"""
Questo modulo definisce una classe base astratta per i modelli SSD (Single Shot MultiBox Detector).

Interfaccia che specifica i metodi che devono essere implementati da qualsiasi modello SSD concreto.

@Author Nicola Guerra
"""
from abc import ABC, abstractmethod

class SSDModel(ABC):
    """
    Classe base astratta per i modelli SSD.

    Metodi
    -------
        load_model() -> None
            Metodo astratto
            Carica il modello SSD.
        load_utils() -> None
            Metodo astratto
            Carica le utility per il modello SSD.
    """

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_utils(self):
        pass
