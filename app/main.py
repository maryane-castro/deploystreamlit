from pipelines.pipeline_nlp import PipelineNLP
import cv2

pipelineNLP = PipelineNLP(ocr_threshold=0.6, similarity_threshold=0.2)
imagem_local = cv2.imread('34f31b3_76e1368d5ca110d7_page9.jpg')
pipelineNLP.set_input('pdi_results.json', imagem_local)
aqui = pipelineNLP.run('intermediate_result.json', 'final_result.json')
print(aqui)