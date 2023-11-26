from torch import argmax, cuda
from transformers import BertTokenizerFast, BertForTokenClassification
import re


class NamedEntityRecognition:
    def __init__(self, model_path='bd_utils/model_ner'):
        """
        Classe para aplicação do NER.

        Args:
            model_path (str): caminho para o diretório com os arquivos do modelo do NER
        """
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.MAX_LEN = 128
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.labels_to_ids = {'B-bra': 1,
                              'B-cat': 2,
                              'B-tip': 5,
                              'B-unit': 3,
                              'I-bra': 4,
                              'I-cat': 7,
                              'I-tip': 6,
                              'I-unit': 8,
                              'B-sep': 9,
                              'O': 0}

        self.ids_to_labels = {j: i for i, j in self.labels_to_ids.items()}

        self.model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(self.labels_to_ids))
        self.model.to(self.device)

    def prepare_sentence(self, sentence: str) -> str:
        """
        Método para preparar a sentença, mudando para caixa baixa e dividindo as vírgulas e barras.

        Args:
            sentence (str): texto a ser processado.

        Returns:
            str: texto recebido após o processamento aplicado.
        """
        sentence = re.sub(r'(?<=\d),(?=\d)', '.', sentence)
        sentence = sentence.lower()
        sentence = sentence.replace(',', ' , ')
        if 'c/' not in sentence and 'c /' not in sentence:
            sentence = sentence.replace('/', ' / ')
        sentence = sentence.replace('\\', ' \ ')

        return sentence

    def predict(self, sentence):
        """
        Método aplicar o reconhecimento das entidades presentes na sentença recebida. Preferencialmente,
        o método prepare_sentence deve ser chamado primeiro.

        Args:
            sentence (str): texto a ser reconhecido.

        Returns:
            list: entidades reconhecidas do texto recebido.
        """
        inputs = self.tokenizer(sentence.split(),
                                is_split_into_words=True,
                                return_offsets_mapping=True,
                                padding='max_length',
                                truncation=True,
                                max_length=self.MAX_LEN,
                                return_tensors="pt")

        # move to gpu
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        # forward pass
        outputs = self.model(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, self.model.num_labels)
        flattened_predictions = argmax(active_logits, axis=1)

        tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [self.ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
            # only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
            else:
                continue

        return prediction
