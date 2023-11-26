import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np


class ProductSearch:
    def __init__(self, database_csv: str):
        """
        Classe para realizar pesquisa de produtos em um banco de dados.

        Args:
            database_csv (str): caminho para o arquivo CSV contendo o banco de dados.
        """
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
        """
        Método privado para carregar o banco de dados e aplicar processamentos.

        Args:
            database_csv (str): caminho para o arquivo CSV contendo o banco de dados.

        Returns:
            DataFrame: banco de dados processado.
        """
        bd = pd.read_csv(database_csv, sep=";")

        bd['description'] = ' ' + bd['description'].astype(str)
        bd['description'] = bd['description'].str.lower()

        bd['description'] = bd['description'].apply(unidecode)
        bd['description'] = bd['description'].str.replace(r'\b\w{1,2}\b', '', regex=True)

        return bd

    def __reduce_bd_by_ocr_findeds(self, list_words_finded_by_ocr: list):
        """
        Método privado para filtrar o banco de dados baseado no produto a ser pesquisado.

        Args:
            list_words_finded_by_ocr (list): lista contendo as palavras pré-processadas da descrição do produto.

        Returns:
            list: lista contendo as descrições dos produtos do banco de dados filtrado.
            list: lista contendo os IDs dos produtos do banco de dados filtrado.
        """
        pattern = ' | '.join(list_words_finded_by_ocr)
        aux = self.database['description'].str.extract(r'(' + pattern + ')')
        aux = self.database.iloc[aux.dropna().index.values]

        return list(aux.description), list(aux.id)

    def __find_ocr_string_product_in_bd_by_similarity(self, processed_bd: list, ocr_findeds: list):
        """
        Método privado para computar a similaridade entre um produto e outros produtos do banco de dados.

        Args:
            processed_bd (list): lista contendo as descrições dos produtos do banco de dados.
            ocr_findeds (list): lista contendo a descrição do produto a ser pesquisado

        Returns:
            ndarray: matriz de similaridade entre o produto pesquisado e os produtos do banco de dados.
        """
        processed_bd.extend(ocr_findeds)
        tfidf_vectorizer = TfidfVectorizer(stop_words=self.stop_words)  # stopwords.words('portuguese'))
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_bd)
        cossine_sim = cosine_similarity(tfidf_matrix)

        return cossine_sim[-1, :-1]

    def __preprocess_product_description(self, product_description: str):
        """
        Método privado para aplicar pré-processamento na descrição de um produto.

        Args:
            product_description (str): descrição do produto de uma oferta.

        Returns:
            list: lista contendo as palavras pré-processadas da descrição do produto.
        """
        product_description = product_description.split()
        preprocessed_product_description = []
        for word in product_description:
            word = re.sub(r'[^\w\s]', ' ', word)
            word = re.sub(r'\b\w{1,2}\b', '', word)
            word = unidecode(word.lower()).split()
            preprocessed_product_description.extend(word)
        return preprocessed_product_description

    def __call__(self, product_description: str, top_n: int = 1):
        """
        Método para realizar a busca de um produto no banco de dados.

        Args:
            product_description (str): descrição do produto de uma oferta.
            top_n (int): quantidade de produtos mais similares do banco para serem retornados (default=1).

        Returns:
            ndarray: similaridades dos produtos encontrados no banco.
            ndarray: descrições dos produtos encontrados no banco.
            ndarray: IDs dos produtos encontrados no banco.
        """
        product_description = self.__preprocess_product_description(product_description)

        if len(product_description) == 0:
            return [0], [''], ['']

        description_candidates, id_candidates = self.__reduce_bd_by_ocr_findeds(product_description)

        if len(description_candidates) == 0:
            return [0], [''], ['']

        cossine_sim = self.__find_ocr_string_product_in_bd_by_similarity(description_candidates,
                                                                         [' '.join(product_description)])
        idx = (-cossine_sim).argsort()[0:top_n]
        similarities = cossine_sim[idx]
        products = np.array(description_candidates)[idx]
        ids = np.array(id_candidates)[idx]
        return similarities, products, ids
