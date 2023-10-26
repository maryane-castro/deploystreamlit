from models.nlp.models_rec import PaddlePyocrRec, EasyOcrRec, KerasOcrRec, CustomPaddleOcrRec, SimilarityDynamicRec


class PipelineRecognizer:
    def __init__(self, threshold_score=0.6):
        """
        Classe do Pipeline de reconhecimento utilizando modelos de OCR.

        Args:
            threshold_score (float): limiar para determinar se a resposta do OCR pode ser aceita.
        """
        self.description_models_list = [PaddlePyocrRec(), EasyOcrRec(), KerasOcrRec()]
        self.price_models_list = [CustomPaddleOcrRec({
            'rec_model_dir': 'models/nlp/price_model/ocr',
            'rec_char_dict_path': 'models/nlp/price_model/price_dict.txt',
            'rec_image_shape': '3, 32, 100'})]
        self.dynamics_models_list = [SimilarityDynamicRec()]
        self.threshold_score = threshold_score

    def recognize(self, image, recognition_type: str):
        """
        Método para aplicar o OCR em uma determinada imagem. Para cada tipo de texto a ser reconhecido,
        vários modelos podem ser testados em sequência até um que apresente uma resposta que seja aceita.

        Args:
            image (ndarray): imagem para aplicar o OCR.
            recognition_type (str): determina o tipo de modelo a ser aplicado {'description', 'price', 'dynamics'}.

        Returns:
            str: resultado do OCR aplicado na imagem.
        """
        assert recognition_type in ['description', 'price', 'dynamics'], \
            "recognition_type must be one of the following: 'description', 'price', 'dynamics'"

        if recognition_type == 'description':
            models_list = self.description_models_list
        elif recognition_type == 'price':
            models_list = self.price_models_list
        else:
            models_list = self.dynamics_models_list

        for model in models_list:
            text, confidence_score = model.recognize(image)
            if confidence_score >= self.threshold_score:
                return text
        return ''
