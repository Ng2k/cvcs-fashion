"""
La classe utilizza un modello SSD pre-addestrato.
Fornisce metodi per caricare un'immagine, elaborarla e disegnare dei bounding box.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import torch
from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib import patches

class SingleShotDetector():
    """
    Classe usata per rappresentare un Single Shot Detector (SSD) model.

    Usa un modello SSD pre-addestrato da NVIDIA's Deep Learning Examples.
    Fornisce metodi per caricare un'immagine, elaborarla e disegnare dei bounding box.

    Attributi
    ----------
        _CONFIDENCE : float
            Attributo privato
            Threshold di confidenza per la detection degli oggetti.
            Gli oggetti con un punteggio di confidenza inferiore a questa soglia vengono ignorati.
        _image : numpy.ndarray
            Attributo privato
            L'immagine da elaborare, rappresentata come un array NumPy.
        _image_tensor : torch.Tensor
            Attributo privato
            L'immagine da elaborare, rappresentata come un tensore PyTorch.

    Methods
    -------
        _load_image(image_url: str) -> None
            Metodo privato
            Carica un'immagine da un URL o un percorso di file e la prepara per l'elaborazione.
        _run(image_tensor: torch.Tensor) -> list
            Metodo privato
            Esegue il modello SSD sull'immagine data e restituisce i risultati della detection.
        _draw_boxes() -> None
            Metodo privato
            Disegna i bounding box sugli oggetti rilevati nell'immagine.
        detection(image_url: str) -> None
            Metodo pubblico
            Carica un'immagine da un URL o un percorso di file, esegue la detection e disegna i bounding box.
    """

    _CONFIDENCE: float = 0.40

    _image: ndarray
    _image_tensor: torch.Tensor

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ssd_model = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_ssd'
        )
        self.utils = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_ssd_processing_utils',
        )

        self.ssd_model.to(device)
        self.ssd_model.eval()

        self._image = None
        self._image_tensor = None

    def _load_image(self, image_url: str) -> None:
        """ Funzione per il caricamento dell'immagine

        Args:
        ------------------
            image_url (str): url dell'immagine

        Returns:
        ------------------
            data (Dict[str, Union[any, torch.Tensor]]): dizionario con l'immagine e il tensore
        """
        self._image = self.utils.prepare_input(image_url)
        self._image_tensor = self.utils.prepare_tensor([self._image])

    def _run(self, image_tensor: torch.Tensor) -> list:
        """ Funzione per l'esecuzione del modello

        Args:
        ------------------
            image (torch.Tensor): immagine da processare

        Returns:
        ------------------
            detection (list): lista delle detection
        """
        with torch.no_grad():
            detections_batch = self.ssd_model(image_tensor)

        results_per_input = self.utils.decode_results(detections_batch)
        best_results_per_input = [
            self.utils.pick_best(results, self._CONFIDENCE) for results in results_per_input
        ]

        return best_results_per_input

    def _draw_boxes(self) -> None:
        """Funzione per disegnare i bounding box

        Args:
            input_image (torch.Tensor): immagine di input
            result (list): lista dei risultati

        Returns:
            list: lista dei risultati
        """
        best_inputs = self._run(self._image_tensor)

        for image_result in best_inputs:
            _, ax = plt.subplots(1)

            # immagine denormalizzata
            ax.imshow(self._image / 2 + 0.5)

            # Immagine con box
            bboxes, _, confidences = image_result

            for idx, bbox in enumerate(bboxes):
                left, bot, right, top = bbox
                x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(
                    x, y,
                    f"{confidences[idx]*100:.0f}%",
                    bbox={"facecolor": 'white', "alpha": 0.5}
                )

            plt.show()

    def detection(self, image_url: str) -> None:
        """Funzione per la detection

        Args:
        ------------------
            image_url (str): url dell'immagine
        """
        self._load_image(image_url)
        self._draw_boxes()
