from pipelines import pipeline_pdi
from pipelines.pipeline_nlp import PipelineNLP
import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
import zipfile




#download recortes

file_id_recortes = '1xyHv-3Nt-hfn_sMBxuc4k3GPl1ivA4nH'
url = f'https://drive.google.com/uc?id={file_id_recortes}'
destination_recortes = 'models/pdi/'
os.makedirs(os.path.dirname(destination_recortes), exist_ok=True)

file_downloaded_recortes = os.path.exists(os.path.join(destination_recortes, 'recortes.pt'))
if not file_downloaded_recortes:
    gdown.download(url, destination_recortes, quiet=False)
    file_downloaded_recortes = True


#download model_ner

file_id_NER = '121Q3hQRu4bNQH7zKoE6GLQ00jdQfBuXj'
url = f'https://drive.google.com/uc?id={file_id_NER}'
destination_NER = 'bd_utils/'

# Verifica se o diretório de destino existe e cria se não existir
os.makedirs(destination_NER, exist_ok=True)

file_downloaded_NER = os.path.exists(os.path.join(destination_NER, 'model_ner'))

if not file_downloaded_NER:
    # Faz o download do arquivo
    gdown.download(url, os.path.join(destination_NER, 'model_ner'), quiet=False)
    file_downloaded_NER = True

# Verifica se o arquivo foi baixado com sucesso
if file_downloaded_NER:
    # Verifica se o arquivo é um arquivo zip (ou outro formato compactado) e descompacta
    file_path = os.path.join(destination_NER, 'model_ner')
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_NER)
        os.remove(file_path) 




st.title('Macro Entrega 2 - Protótipo PDI e NLP')

uploaded_image = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:

    #pdi
    image = Image.open(uploaded_image).convert("RGB")
    image = np.array(image)
    resultados_pdi, imagem = pipeline_pdi.pipelinePDI(image_path=image, image_name=uploaded_image.name)
    #st.json(resultados_pdi)
    st.image(imagem, caption='Exemplo de Imagem', use_column_width=True)

    #nlp
    pipelineNLP = PipelineNLP(ocr_threshold=0.3, similarity_threshold=0.6)
    pipelineNLP.set_input('pdi_results.json', image)
    resultados_pdi = pipelineNLP.run('intermediate_result.json', 'final_result.json')
    st.json(resultados_pdi)