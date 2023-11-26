import numpy as np


class ProcessComplexDescriptions:
    def __init__(self):
        """
        Classe para tratamento dos casos das descrições dos produtos baseado nas anotações do NER.
        """
        pass

    def __processing_description_type_2(self, ner_annotation: list, ocr_sentence: list):
        """
        Método privado que processa descrições complexas que variam apenas dois ou mais tipos de
        anotação (B-bra, B-tip, B-unit or B-cat).
        Ex: 'manga tommy ou tangerina murcute kg'
            ['B-cat', 'B-tip', 'B-sep', 'B-cat', 'B-tip', 'B-unit']

        Args:
            ner_annotation (list): lista de anotações do NER.
            ocr_sentence (list): lista das strings do OCR correspondentes as anotações do ner_annotation.

        Returns:
            list: lista contendo os produtos do ocr_sentence de acordo com as anotações do NER.
        """
        ocr_sentence = np.array(ocr_sentence)
        ner_annotation = np.array(ner_annotation)

        final_sentences = {}

        values, counts = np.unique(ner_annotation, return_counts=True)

        sep_count = 0
        final_sentences[sep_count] = []
        single_locals = []

        for val in values:

            if 'sep' in val:  # remove separate setences
                sep_idx = np.where(ner_annotation == val)[0]

            if 'B-' in val:  # just consider annotatios with B- type
                if 'sep' not in val:
                    locals = np.where(ner_annotation == val)[0]

                    if len(locals) == 1:
                        single_locals.extend(locals)

        repeated_setences = np.delete(ocr_sentence, single_locals)
        ner_repeated = np.delete(ner_annotation, single_locals)

        for ner_annot, ocr_sent in zip(ner_repeated, repeated_setences):

            if 'sep' in ner_annot:
                final_sentences[sep_count].extend(ocr_sentence[single_locals])
                sep_count += 1
                final_sentences[sep_count] = []
            else:
                final_sentences[sep_count].append(ocr_sent)

        final_sentences[sep_count].extend(ocr_sentence[single_locals])

        result = []
        for sentence in final_sentences.values():
            final_sentence = ' '.join(sentence)
            result.append(final_sentence)
        return result

    def __processing_description_type_1(self, ner_annotation: list, ocr_sentence: list):
        """
        Método privado que processa descrições complexas que variam apenas um tipo de
        anotação (B-bra, B-tip, B-unit or B-cat).
        Ex: 'arroz branco ou parbolizado tio márcio 1kg'
            ['B-cat', 'B-tip', 'B-sep', 'B-tip', 'B-bra', 'I-bra', 'B-unit']

            'arroz branco tio mácio 1kg / 5kg'
            ['B-cat', 'B-tip', 'B-bra', 'I-bra', 'B-unit', 'B-sep', 'B-unit']

            'vinho ou catuaba são brás 1l'
            ['B-cat', 'B-sep', 'B-cat', 'B-bra', 'I-bra', 'B-unit']

        Args:
            ner_annotation (list): lista de anotações do NER.
            ocr_sentence (list): lista das strings do OCR correspondentes as anotações do ner_annotation.

        Returns:
            list: lista contendo os produtos do ocr_sentence de acordo com as anotações do NER.
        """
        ocr_sentence = np.array(ocr_sentence)
        ner_annotation = np.array(ner_annotation)

        values, counts = np.unique(ner_annotation, return_counts=True)
        for val in values:

            if 'sep' in val:  # remove separate setences
                sep_idx = np.where(ner_annotation == val)[0]

            if 'B-' in val:  # just consider annotatios with B- type
                if 'sep' not in val:
                    locals = np.where(ner_annotation == val)[0]

                    if len(locals) > 1:
                        mult_locals = locals

        commons_setences = np.delete(ocr_sentence, np.concatenate((mult_locals, sep_idx), axis=0), axis=0)
        final_sentences = {}

        for idx in mult_locals:
            final_sentences[idx] = commons_setences
            final_sentences[idx] = np.insert(final_sentences[idx], 0, ocr_sentence[idx])

        result = []
        for sentence in final_sentences.values():
            result.append(' '.join(list(sentence)))
        return result

    def __find_type_patter(self, ner_annotation: list) -> str:
        """
        Método privado para encontrar o tipo de sentença do OCR. Existem n tipos, onde cada um se refere à quantidade
        de anotações B.

        Args:
            ner_annotation (list): lista de anotações do NER.

        Returns:
            str: indicação de qual é o tipo de sentença do OCR.
        """
        values, counts = np.unique(np.array(ner_annotation), return_counts=True)
        diff_sentence = {}

        for val, cou in zip(values, counts):
            if val == 'B-sep':  # desconsiderar o separador
                continue
            if cou >= 2:
                diff_sentence[val] = cou

        if len(diff_sentence) == 1:  # tipo1: variando somente um tipo de anotação
            return 'type 1'
        elif len(diff_sentence) > 1:
            return 'type 2'
        else:
            return 'type error'

    def __process_sentence(self, ocr_sentence: str, ner_annotation: list):
        """
        Método privado para separar a sentença baseado na anotação do NER.

        Args:
            ocr_sentence (str): string computada pelo OCR.
            ner_annotation (list): lista de anotações do NER.

        Returns:
            list: lista das strings do OCR correspondentes as anotações do ner_annotation.
            list: lista de anotações do NER após o processamento.
        """
        ocr_sentence = np.array(ocr_sentence.split())
        ner_annotation.append('B-end')
        ner_annotation = np.array(ner_annotation)

        aux = []
        aux_annot = ''
        idx_aux = []

        new_ocr_sentece = []
        new_ner_annotation = []

        for annot_idx, annot in enumerate(ner_annotation):

            if annot == 'B-end':
                break

            if 'B-' in annot:
                if len(aux) >= 1:  # there is something in aux and mount complete sentence
                    new_ocr_sentece.append(' '.join(ocr_sentence[idx_aux]))
                    new_ner_annotation.append(aux[0])

                    aux = []
                    idx_aux = []

                    aux_annot = annot[2:]
                    aux.append(annot)
                    idx_aux.append(annot_idx)
                else:

                    aux_annot = annot[2:]
                    aux.append(annot)
                    idx_aux.append(annot_idx)

            elif 'I-' in annot:
                if annot[2:] == aux_annot:
                    aux.append(annot)
                    idx_aux.append(annot_idx)
                else:
                    annot = annot.replace('I-', 'B-')
                    new_ocr_sentece.append(' '.join(ocr_sentence[idx_aux]))

                    if len(aux) == 0:
                        new_ner_annotation.append(annot)
                    else:
                        new_ner_annotation.append(aux[0])

                    aux = []
                    idx_aux = []

                    aux_annot = annot[2:]
                    aux.append(annot)
                    idx_aux.append(annot_idx)

                if len(aux) < 1:
                    annot = annot.replace('I-', 'B-')
                    aux_annot = annot[2:]
                    aux.append(annot)
                    idx_aux.append(annot_idx)

            if annot_idx == len(ner_annotation) - 2:
                new_ocr_sentece.append(' '.join(ocr_sentence[idx_aux]))
                new_ner_annotation.append(aux[0])

        return new_ocr_sentece, new_ner_annotation

    def mount_descriptions(self, result_ner: list, ocr_sentence: str):

        """
        Método para detectar os produtos das descrições baseados nas anotações do NER, de acordo com o seguinte
        padrão de descrição de produto: categoria (cat) + marca (bra) + tipo (tip) + unidade de medida (unit).

        Args:
            result_ner (list): lista de anotações do NER.
            ocr_sentence (str): string computada pelo OCR.

        Returns:
            list: lista contendo os produtos do ocr_sentence de acordo com as anotações do NER.
        """
        new_ocr_sentece, new_ner_annotation = self.__process_sentence(ocr_sentence, result_ner)
        type_sentence = self.__find_type_patter(new_ner_annotation)

        if type_sentence == 'type 1':  # varying only 1 type of sentence B-
            return self.__processing_description_type_1(new_ner_annotation, new_ocr_sentece)
        elif type_sentence == 'type 2':
            return self.__processing_description_type_2(new_ner_annotation, new_ocr_sentece)
        else:
            return [ocr_sentence]
