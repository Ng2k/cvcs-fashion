"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
class TextProcessor:
    """
    Classe TextProcessor per il pre-processing del testo
    """
    def __init__(self, tokenizer, device='cpu'):
        self.tokenizer = tokenizer
        self.device = device

    def tokenize_prompts(self, prompts: list):
        """Tokenizza i prompt di testo

        Parametri:
        -------
            prompts (list): lista di prompt di testo

        Returns:
        -------
            _type_: _description_
        """
        tokenized_prompts = self.tokenizer(prompts).to(self.device)
        return tokenized_prompts