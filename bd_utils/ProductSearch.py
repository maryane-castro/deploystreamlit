import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np


class ProductSearch:
    def __init__(self, database_csv: str):
        self.database = self.__load_bd(database_csv)
        self.stop_words = ['ml', 'und', 'g', '9', 'l', 'kg', 'unid', 'kilo', 'kilos', 'gramas', 'grama', 'und', 'unidades',
                      '%', 'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'e', 'com', 'nao', 'uma', 'os',
                      'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem',
                      'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também',
                      'so', 'pelo', 'pela', 'ate', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos',
                      'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 'tinha', 'foram', 'essa',
                      'num', 'nem', 'suas', 'meu', 'as', 'minha', 'têm', 'numa', 'pelos', 'elas', 'havia', 'seja',
                      'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse',
                      'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas',
                      'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele',
                      'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'esta', 'estamos', 'estao', 'estive',
                      'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estiveramos',
                      'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver',
                      'estivermos', 'estiverem', 'hei', 'ha', 'havemos', 'hao', 'houve', 'houvemos', 'houveram',
                      'houvera', 'houveramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem',
                      'houver', 'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos', 'houverao', 'houveria',
                      'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'eramos', 'eram', 'fui', 'foi', 'fomos',
                      'foram', 'fora', 'foramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fossemos', 'fossem', 'for',
                      'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho',
                      'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram',
                      'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem',
                      'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos',
                      'teriam']

    def __load_bd(self, database_csv: str):
        bd = pd.read_csv(database_csv, sep=";")

        bd['description'] = ' ' + bd['description'].astype(str)
        bd['description'] = bd['description'].str.lower()

        bd['description'] = bd['description'].apply(unidecode)
        bd['description'] = bd['description'].str.replace(r'\b\w{1,2}\b', '', regex=True)

        return bd

    def __reduce_bd_by_ocr_findeds(self, list_words_finded_by_ocr: list):
        pattern = ' | '.join(list_words_finded_by_ocr)
        aux = self.database['description'].str.extract(r'(' + pattern + ')')
        aux = self.database.iloc[aux.dropna().index.values]

        return list(aux.description), list(aux.id)

    def __find_ocr_string_product_in_bd_by_similarity(self, processed_bd: list, ocr_findeds: list):
        processed_bd.extend(ocr_findeds)

        tfidf_vectorizer = TfidfVectorizer(stop_words=self.stop_words)  # stopwords.words('portuguese'))
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_bd)
        cossine_sim = cosine_similarity(tfidf_matrix)

        return cossine_sim

    def __preprocess_product_description(self, product_description: str):
        product_description = product_description.split()
        preprocessed_product_description = []
        for word in product_description:
            word = re.sub(r'[^\w\s]', ' ', word)
            word = re.sub(r'\b\w{1,2}\b', '', word)
            word = unidecode(word.lower()).split()
            preprocessed_product_description.extend(word)
        return preprocessed_product_description

    def __call__(self, product_description: str, top_n: int = 1):
        product_description = self.__preprocess_product_description(product_description)
        description_candidates, id_candidates = self.__reduce_bd_by_ocr_findeds(product_description)
        if len(description_candidates) == 0:
            return [0], [''], ['']
        cossine_sim = self.__find_ocr_string_product_in_bd_by_similarity(description_candidates,
                                                                         [' '.join(product_description)])
        idx = (-cossine_sim[-1]).argsort()[1:top_n+1]
        similarities = cossine_sim[-1][idx]
        products = np.array(description_candidates)[idx]
        ids = np.array(id_candidates)[idx]
        return similarities, products, ids
