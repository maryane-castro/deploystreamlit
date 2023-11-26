from bd_utils.ProductSearch import ProductSearch
from bd_utils.NamedEntityRecognition import NamedEntityRecognition
from bd_utils.ProcessComplexDescriptions import ProcessComplexDescriptions


class OfferCaseHandling:
    def __init__(self, database_csv: str):
        """
        Classe para pesquisar os itens encontrados pelo OCR em um banco de dados e fazer
        o tratamento para os possíveis casos:
            - 1 oferta e 1 preço;
            - 1 oferta e n preços;
            - n ofertas e 1 preço;
            - n ofertas e n preços.

        Args:
            database_csv (str): caminho para o arquivo CSV contendo o banco de dados
        """
        self.product_search = ProductSearch(database_csv)
        self.ner = NamedEntityRecognition()
        self.process_complex_descriptions = ProcessComplexDescriptions()

    def __treating_1_offer_1_price(self, offers, prices, dynamics):
        """
        Método privado para tratar o caso de 1 oferta e 1 preço.

        Args:
            offers (list): lista contendo as ofertas detectadas nas descrições.
            prices (list): lista contendo os preços detectados pelo OCR.
            dynamics (list): lista contendo as dinâmicas detectadas pelo OCR.

        Returns:
            list: lista de dicionários contendo as informações detectadas pelo pipeline de NLP de
            cada oferta recebida como argumento.
        """
        ids, products, similarities = self.__find_offers_in_database(offers)
        return [{'product_id': ids[0],
                 'description': products[0],
                 'similarity': similarities[0],
                 'price': prices[0],
                 'dynamic': dynamics[0]}]

    def __treating_1_offer_n_price(self, offers, prices, dynamics):
        """
        Método privado para tratar o caso de 1 oferta e n preços.

        Args:
            offers (list): lista contendo as ofertas detectadas nas descrições.
            prices (list): lista contendo os preços detectados pelo OCR.
            dynamics (list): lista contendo as dinâmicas detectadas pelo OCR.

        Returns:
            list: lista de dicionários contendo as informações detectadas pelo pipeline de NLP de
            cada oferta recebida como argumento.
        """
        ids, products, similarities = self.__find_offers_in_database(offers)
        return [{'product_id': ids[0],
                 'description': products[0],
                 'similarity': similarities[0],
                 'price': price,
                 'dynamic': dynamic} for price, dynamic in zip(prices, dynamics)]

    def __treating_n_offer_1_price(self, offers, prices, dynamics):
        """
        Método privado para tratar o caso de n ofertas e 1 preço.

        Args:
            offers (list): lista contendo as ofertas detectadas nas descrições.
            prices (list): lista contendo os preços detectados pelo OCR.
            dynamics (list): lista contendo as dinâmicas detectadas pelo OCR.

        Returns:
            list: lista de dicionários contendo as informações detectadas pelo pipeline de NLP de
            cada oferta recebida como argumento.
        """
        ids, products, similarities = self.__find_offers_in_database(offers)
        return [{'product_id': product_id,
                 'description': product,
                 'similarity': similarity,
                 'price': prices[0],
                 'dynamic': dynamics[0]} for product_id, product, similarity in zip(ids, products, similarities)]

    def __treating_n_offer_n_price(self, offers, prices, dynamics):
        """
        Método privado para tratar o caso de n ofertas e n preços.

        Args:
            offers (list): lista contendo as ofertas detectadas nas descrições.
            prices (list): lista contendo os preços detectados pelo OCR.
            dynamics (list): lista contendo as dinâmicas detectadas pelo OCR.

        Returns:
            list: lista de dicionários contendo as informações detectadas pelo pipeline de NLP de
            cada oferta recebida como argumento.
        """
        ids, products, similarities = self.__find_offers_in_database(offers)
        data = []
        for product_id, product, similarity in zip(ids, products, similarities):
            data.extend([{'product_id': product_id,
                          'description': product,
                          'similarity': similarity,
                          'price': price,
                          'dynamic': dynamic} for price, dynamic in zip(prices, dynamics)])
        return data

    def __find_offers_in_database(self, offers):
        """
        Método privado para pesquisar as ofertas no banco de dados.

        Args:
            offers (list): lista contendo as ofertas detectadas nas descrições.

        Returns:
            ndarray: IDs dos produtos encontrados no banco.
            ndarray: descrições dos produtos encontrados no banco.
            ndarray: similaridades dos produtos encontrados no banco.
        """
        similarities = []
        products = []
        ids = []
        for offer in offers:
            if len(offer) <= 4:
                products.append(None)
                ids.append(None)
                similarities.append(0)
            else:
                similarity, product, product_id = self.product_search(offer)
                products.append(product[0])
                ids.append(product_id[0])
                similarities.append(similarity[0])
        return ids, products, similarities

    def __get_offers_from_description(self, descriptions):
        """
        Método privado para detectar as ofertas presentes em descrições por meio de
        Reconhecimento de Entidade Mencionada (NER).

        Args:
            descriptions (list): lista contendo descrições de produtos detectadas pelo OCR.

        Returns:
            list: lista contendo as ofertas detectadas nas descrições.
        """
        offers = []
        for description in descriptions:
            description_aux = self.ner.prepare_sentence(description)
            is_complex = ' , ' in description_aux or ' / ' in description_aux or ' ou ' in description_aux
            if is_complex:
                ner_annotation = self.ner.predict(description_aux)
                if 'sep' not in ner_annotation:
                    offers.append(description)
                else:
                    all_product_descriptions = self.process_complex_descriptions.mount_descriptions(ner_annotation, description_aux)
                    offers.extend(all_product_descriptions)
            else:
                offers.append(description)
        return offers

    def __call__(self, descriptions: list, prices: list, dynamics: list):
        """
        Método para realizar o tratamento completo da saída do OCR.

        Args:
            descriptions (list): lista contendo descrições de produtos detectadas pelo OCR.
            prices (list): lista contendo os preços detectados pelo OCR.
            dynamics (list): lista contendo as dinâmicas detectadas pelo OCR.

        Returns:
            list: lista de dicionários contendo as informações detectadas pelo pipeline de NLP de
            cada oferta recebida como argumento.
        """
        assert len(prices) == len(dynamics), "Lists prices and dynamics must have the same length"
        offers = self.__get_offers_from_description(descriptions)
        if len(offers) == 1 and len(prices) == 1:
            return self.__treating_1_offer_1_price(offers, prices, dynamics)
        elif len(offers) == 1 and len(prices) > 1:
            return self.__treating_1_offer_n_price(offers, prices, dynamics)
        elif len(offers) > 1 and len(prices) == 1:
            return self.__treating_n_offer_1_price(offers, prices, dynamics)
        elif len(offers) > 1 and len(prices) > 1:
            return self.__treating_n_offer_n_price(offers, prices, dynamics)
        else:
            return [{'similarity': 0}]
