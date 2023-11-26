import numpy as np
from utils.json_utils import read_pdi_json, save_nlp_intermediate_json, save_nlp_final_json
from utils.api_requests import get_leaflet_image
from models.nlp.PipelineRecognizer import PipelineRecognizer
from bd_utils.OfferCaseHandling import OfferCaseHandling
from utils.image_utils import cut_image


class PipelineNLP:
    def __init__(self, ocr_threshold, similarity_threshold, path_bd='bd_utils/leaflet_items.csv'):
        """
        Classe do pipeline completo de NLP.

        Args:
            ocr_threshold (float): limiar para determinar se a resposta do OCR pode ser aceita.
            similarity_threshold (float): limiar de similaridade dos itens pesquisados no banco.
            path_bd (str): caminho para o arquivo CSV contendo o banco de dados.
        """
        self.leaflet_id = None
        self.date = None
        self.page_number = None
        self.leaflet_image = None
        self.offers = None
        self.similarity_threshold = similarity_threshold
        self.pipeline_recognizer = PipelineRecognizer(threshold_score=ocr_threshold)
        self.offer_case_handling = OfferCaseHandling(path_bd)

    def set_input(self, path_json):
        """
        Método para receber o JSON de saída do PDI, definindo os atributos da classe.

        Args:
            path_json (str): caminho para o JSON de saída do PDI.

        Returns:
            None.
        """
        self.leaflet_id, self.date, self.page_number, self.offers = read_pdi_json(path_json)
        self.leaflet_image = get_leaflet_image(self.date, self.leaflet_id, self.page_number)

    def __get_category_result(self, bboxes, recognition_type):
        """
        Método privado para aplicar o OCR em imagens de bounding boxes do encarte.

        Args:
            bboxes (list): lista de dicionários contendo bounding boxes no seguinte formato:
            {'x1': x1, 'x2: x2, 'y1': y1, 'y2': y2}.
            recognition_type (str): determina o tipo de modelo a ser aplicado {'description', 'price', 'dynamics'}.

        Returns:
            list: lista contendo os resultados do OCR para cada bounding box.
        """
        result = []
        for bbox in bboxes:
            crop_image = cut_image(self.leaflet_image, **bbox)
            result.append(self.pipeline_recognizer.recognize(crop_image, recognition_type))
        return result

    def __get_dynamic_result(self, bboxes, price_list):
        """
        Método privado para aplicar o OCR em imagens de bounding boxes do encarte para a detecção
        de dinâmica. Internamente é realizado um tratamento para que o número de dinâmicas seja
        igual ao de preços.

        Args:
            bboxes (list): lista de dicionários contendo bounding boxes no seguinte formato:
            {'x1': x1, 'x2: x2, 'y1': y1, 'y2': y2}.
            price_list (list): lista contendo os preços detectados pelo OCR.

        Returns:
            list: lista contendo as dinâmicas detectadas.
        """
        dynamic_list = ['OFERTA_DE_POR'] * len(price_list)
        if len(bboxes[0]) != 0:
            price_argsort = np.argsort([float(i.replace(',', '.')) for i in price_list])
            dynamic_list_aux = self.__get_category_result(bboxes, recognition_type='dynamics')
            for i, dynamic in enumerate(dynamic_list_aux):
                dynamic_list[price_argsort[i]] = dynamic
        return dynamic_list

    def __get_ocr_result(self, offer):
        """
        Método privado para computar os resultados do OCR de uma determinada oferta detectada pelo PDI.

        Args:
            offer (dict): dicionário contendo as informações de uma oferta detectada no encarte pelo PDI.

        Returns:
            dict: dicionário contendo as descrições, preços e dinâmicas de uma oferta detectados pelo OCR.
        """
        description_list = self.__get_category_result(offer['description'], recognition_type='description')
        price_list = self.__get_category_result(offer['price'], recognition_type='price')
        dynamic_list = self.__get_dynamic_result(offer['dynamics'], price_list)
        return {'descriptions': description_list,
                'prices': price_list,
                'dynamics': dynamic_list}

    def run(self, path_intermediate_json, path_final_json):
        """
        Método para aplicar o pipeline completo de NLP:

        PDI JSON -> |OCR| -> |BD SEARCH| -> |OFFER CASE HANDLING| -> FINAL AND INTERMEDIATE JSON

        Esse método deve ser chamado depois do método set_input()

        Args:
            path_intermediate_json (str): caminho para salvar o JSON intermediário de saída do NLP.
            path_final_json (str): caminho para salvar o JSON final de saída do NLP.

        Returns:
            None.
        """
        assert self.offers is not None, "You must call set_input() method first"

        ocr_results = [self.__get_ocr_result(offer) for offer in self.offers]

        save_nlp_intermediate_json(path_intermediate_json, self.leaflet_id, self.page_number, self.offers, ocr_results)

        bd_results = [self.offer_case_handling(**r) for r in ocr_results]

        save_nlp_final_json(path_final_json, bd_results, self.leaflet_id, self.page_number, self.offers, self.similarity_threshold)
