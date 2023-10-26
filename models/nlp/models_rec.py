from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import init_args
from paddleocr.tools.infer.predict_rec import TextRecognizer
import easyocr
import keras_ocr
import numpy as np
from difflib import SequenceMatcher


class ModelRecInterface:
    """
    Interface que determina o comportamento de um modelo de OCR.
    """
    def recognize(self, image: np.ndarray) -> (str, float):
        """
        Método para realizar a reconhecimento do texto de interesse em uma imagem.

        Args:
            image (ndarray): imagem para aplicar o OCR.

        Returns:
            str: texto detectado pelo OCR.
            float: pontuação de confiância da resposta do modelo.
        """
        pass


class PaddlePyocrRec(ModelRecInterface):
    def __init__(self):
        """
        Classe do PyOCR da Paddle para o reconhecimento de texto.
        """
        self.recognizer = PaddleOCR(use_angle_cls=True,
                        lang='pt',
                        det=False,
                        )

    def recognize(self, image):
        """
        Método para realizar a reconhecimento do texto de interesse em uma imagem.

        Args:
            image (ndarray): imagem para aplicar o OCR.

        Returns:
            str: texto detectado pelo OCR.
            float: pontuação de confiância da resposta do modelo.
        """
        result = self.recognizer.ocr(image, cls=True)[0]
        #print('resultado', result)

        if result is not None:
            text = ' '.join([r[1][0] for r in result])
            confidence_score = np.mean([r[1][1] for r in result])
        else:
            text = ''  # Set an empty text or handle it as needed
            confidence_score = 0.0  # Set a default confidence score

        return text, confidence_score
    


class EasyOcrRec(ModelRecInterface):
    def __init__(self):
        """
        Classe do EasyOCR para o reconhecimento de texto.
        """
        self.recognizer = easyocr.Reader(['pt'])

    def recognize(self, image):
        """
        Método para realizar a reconhecimento do texto de interesse em uma imagem.

        Args:
            image (ndarray): imagem para aplicar o OCR.

        Returns:
            str: texto detectado pelo OCR.
            float: pontuação de confiância da resposta do modelo.
        """
        result = self.recognizer.readtext(image)
        text = ' '.join([r[1] for r in result])
        confidence_score = np.mean([r[2] for r in result])
        return text, confidence_score


class KerasOcrRec(ModelRecInterface):
    def __init__(self):
        """
        Classe do KerasOCR para o reconhecimento de texto.
        """
        self.recognizer = keras_ocr.recognition.Recognizer()

    def recognize(self, image):
        """
        Método para realizar a reconhecimento do texto de interesse em uma imagem.

        Args:
            image (ndarray): imagem para aplicar o OCR.

        Returns:
            str: texto detectado pelo OCR.
            float: pontuação de confiância da resposta do modelo.
        """
        text = self.recognizer.recognize(image)
        return text, 0.7 # NÃO TEM COMO PEGAR O SCORE?


class CustomPaddleOcrRec(ModelRecInterface):
    def __init__(self, model_args):
        """
        Classe de um modelo local da Paddle para o reconhecimento de texto.

        Args:
            model_args (dict): dicionário contendo os argumentos necessário para o carregamento do modelo.
        """
        self.recognizer = self.__load_model(model_args)

    def __load_model(self, model_args):
        """
        Método privado para carregar o modelo da Paddle.

        Args:
            model_args (dict): dicionário contendo os argumentos necessário para o carregamento do modelo.

        Returns:
            TextRecognizer: modelo para OCR.
        """
        args = init_args()
        for arg, value in model_args.items():
            args.set_defaults(**{arg: value})
        rec_model = TextRecognizer(args.parse_args())
        return rec_model

    def recognize(self, image):
        """
        Método para realizar a reconhecimento do texto de interesse em uma imagem.

        Args:
            image (ndarray): imagem para aplicar o OCR.

        Returns:
            str: texto detectado pelo OCR.
            float: pontuação de confiância da resposta do modelo.
        """
        result, _ = self.recognizer([image])
        text = result[0][0]
        confidence_score = result[0][1]
        return text, confidence_score


class SimilarityDynamicRec(ModelRecInterface):
    def __init__(self):
        """
        Classe do modelo para o reconhecimento de dinâmica. Utiliza o PyOCR da Paddle para detecção
        do texto e, então, determina a dinâmica por meio de cálculo de similaridade com uma lista
        de dinâmicas.
        """
        self.dynamics = ['A_PARTIR_DE_X_UNIDADES', 'CARTAO_FIDELIDADE', 'CASHBACK', 'COMBO', 'COMPRE_GANHE',
                         'DESCONTO_APP', 'DESCONTO_NA_2A_UNIDADE', 'DESCONTO_NA_CAIXA_FECHADA', 'PIX',
                         'DESCONTO_NO_CARTAO_PROPRIO', 'LEVE_MAIS_PAGUE_MENOS', 'OFERTA_DE_POR', 'PROGRAMA_FIDELIDADE']
        self.recognizer = PaddlePyocrRec()
        self.similarity_threshold = 0.4

    def recognize(self, image):
        """
        Método para realizar a reconhecimento da dinâmica em uma imagem.

        Args:
            image (ndarray): imagem para aplicar o OCR.

        Returns:
            str: dinãmica detectada pelo OCR e similaridade.
            float: pontuação de confiância da resposta do modelo.
        """
        text, confidence_score = self.recognizer.recognize(image)
        similarities = [SequenceMatcher(None, i, text).ratio() for i in self.dynamics]
        idx = np.argmax(similarities)
        similarity = similarities[idx]
        if similarity >= self.similarity_threshold:
            dynamic = self.dynamics[idx]
            return dynamic, confidence_score
        else:
            return 'OFERTA_DE_POR', confidence_score