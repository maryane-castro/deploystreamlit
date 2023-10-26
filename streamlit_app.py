import json
import cv2
import streamlit as st
from pipeline.PipelinePDI import PipelinePDI
from pipeline.PipelineNLP import PipelineNLP

def execute_pipeline(url_image, date):
    update_mocked_json(url_image, date, 'pdi_mocked.json', overwrite=True)
    
    pipelinePDI = PipelinePDI()
    _, url_json, image = pipelinePDI.set_input('pdi_mocked.json') 
    results = pipelinePDI.detect_offers(url_json, image, date)
    
    with open('final_results_pdi.json', 'w') as json_file:
        json.dump(results, json_file, indent=2)
    pipelineNLP = PipelineNLP(ocr_threshold=0.6, similarity_threshold=0.2)
    pipelineNLP.set_input('final_results_pdi.json')  

    pipelineNLP.run('intermediate_result_nlp.json', 'final_result_nlp.json')

def update_mocked_json(url_image, date, mocked_json_file, overwrite=False):
    if not overwrite:
        with open(mocked_json_file, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = []

    new_entry = {
        "url_image": url_image,
        "date": date
    }

    data.append(new_entry)

    with open(mocked_json_file, 'w') as json_file:
        json.dump(data, json_file, indent=2)

st.title("PDI + NLP")

new_url_image = st.text_input("Nova URL da Imagem")
new_date = st.text_input("Nova Data (no formato YYYY-MM)")

if st.button("Executar"):
    execute_pipeline(new_url_image, new_date)
    st.success("Executado com sucesso!")

    st.write("Resultados Finais:")
    with open('final_result_nlp.json') as nlp_json:
        nlp_results = json.load(nlp_json)
        st.json(nlp_results)
