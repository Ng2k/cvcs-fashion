"""
La classe utilizza un modello SSD pre-addestrato.
Fornisce metodi per caricare un'immagine, elaborarla e disegnare dei bounding box.
"""
import torch
from matplotlib import pyplot as plt
from matplotlib import patches

class SingleShotDetector():
    """Classe per la creazione del modello Single Shot Detector
    """

    CONFIDENCE = 0.40

    def __init__(self):
        self.ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        self.utils = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_ssd_processing_utils'
        )

        self.ssd_model.to('cuda')
        self.ssd_model.eval()

    def _load_image(self, image_url: str) -> torch.Tensor:
        """ Funzione per il caricamento dell'immagine

        Args:
        ------------------
            image_url (str): url dell'immagine

        Returns:
        ------------------
            image (torch.Tensor): immagine caricata
        """
        image = self.utils.prepare_input(image_url)
        return image

    def _run(self, image: torch.Tensor):
        """ Funzione per l'esecuzione del modello

        Args:
        ------------------
            image (torch.Tensor): immagine da processare

        Returns:
        ------------------
            detection (list): lista delle detection
        """
        with torch.no_grad():
            detections_batch = self.ssd_model(image)

        results_per_input = self.utils.decode_results(detections_batch)
        best_results_per_input = [
            self.utils.pick_best(results, self.CONFIDENCE) for results in results_per_input
        ]

        return best_results_per_input

    def _draw_boxes(self, input_image: torch.Tensor) -> None:
        """Funzione per disegnare i bounding box

        Args:
            input_image (torch.Tensor): immagine di input
            result (list): lista dei risultati

        Returns:
            list: lista dei risultati
        """
        best_inputs = self._run(input_image)

        for image_result in best_inputs:
            _, ax = plt.subplots(1)

            # immagine denormalizzata
            ax.imshow(input_image / 2 + 0.5)

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
        image = self._load_image(image_url)
        self._draw_boxes(image)
