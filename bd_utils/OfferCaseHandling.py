from bd_utils.ProductSearch import ProductSearch


class OfferCaseHandling:
    def __init__(self, database_csv: str):
        self.product_search = ProductSearch(database_csv)

    def __treating_1_offer_1_price(self, offers, prices, dynamics):
        ids, products, similarities = self.__find_offers_in_database(offers)
        return [{'product_id': ids[0],
                 'description': products[0],
                 'similarity': similarities[0],
                 'price': prices[0],
                 'dynamic': dynamics[0]}]

    def __treating_1_offer_n_price(self, offers, prices, dynamics):
        ids, products, similarities = self.__find_offers_in_database(offers)
        return [{'product_id': ids[0],
                 'description': products[0],
                 'similarity': similarities[0],
                 'price': price,
                 'dynamic': dynamic} for price, dynamic in zip(prices, dynamics)]

    def __treating_n_offer_1_price(self, offers, prices, dynamics):
        ids, products, similarities = self.__find_offers_in_database(offers)
        return [{'product_id': product_id,
                 'description': product,
                 'similarity': similarity,
                 'price': prices[0],
                 'dynamic': dynamics[0]} for product_id, product, similarity in zip(ids, products, similarities)]

    def __treating_n_offer_n_price(self, offers, prices, dynamics):
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
        similarities = []
        products = []
        ids = []
        for offer in offers:
            if len(offer) <= 4:
                continue
            similarity, product, product_id = self.product_search(offer)
            products.append(product[0])
            ids.append(product_id[0])
            similarities.append(similarity[0])
        return ids, products, similarities

    def __get_offers_from_description(self, description):
        seps = [' ou ', ' | ', ' e ']
        offers = description.copy()
        for sep in seps:
            offers_aux = []
            for d in offers:
                offers_aux.extend(d.split(sep))
            offers = offers_aux
        return offers

    def __call__(self, descriptions: list, prices: list, dynamics: list):
        assert len(prices) == len(dynamics), "Lists prices and dynamics must have the same length"
        offers = self.__get_offers_from_description(descriptions)
        if len(offers) == 1 and len(prices) == 1:
            return self.__treating_1_offer_1_price(offers, prices, dynamics)
        elif len(offers) == 1 and len(prices) > 1:
            return self.__treating_1_offer_n_price(offers, prices, dynamics)
        elif len(offers) > 1 and len(prices) == 1:
            return self.__treating_n_offer_1_price(offers, prices, dynamics)
        else:
            return self.__treating_n_offer_n_price(offers, prices, dynamics)
