import json
import cv2



# # PDI
# from pipeline.PipelinePDI import PipelinePDI
# pipelinePDI = PipelinePDI()
# date, url_json, image = pipelinePDI.set_input('pdi_mocked.json')
# #cv2.imwrite('noe.jpg', image)
# results = pipelinePDI.detect_offers(url_json, image, date)
# with open('final_results_pdi.json', 'w') as json_file:
#     json.dump(results, json_file, indent=2)








# NLP
from pipeline.PipelineNLP import PipelineNLP
pipelineNLP = PipelineNLP(ocr_threshold=0.6, similarity_threshold=0.2)
pipelineNLP.set_input('final_results_pdi.json')

pipelineNLP.run('intermediate_result_nlp.json', 'final_result_nlp.json')
