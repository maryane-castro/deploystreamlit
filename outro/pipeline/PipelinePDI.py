import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
import json
from utils.api_requests import get_leaflet_image_pdi


class PipelinePDI:
    def __init__(self):
        self.model_offer = YOLO('./models/pdi/ofetas.pt')
        self.model_recorte = YOLO('./models/pdi/recortes.pt')


    def set_input(self, json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        url_json, date = data[0].get('url_image', None), data[0].get('date', None)
        parts = url_json.split('_')
        leaflet_id = parts[1]
        number_page = parts[2].split('page')[1]
        image_api = get_leaflet_image_pdi(f"https://s3.amazonaws.com/encartes/{date}/{leaflet_id}-{number_page}")
        return date, url_json, image_api
    

    def json_pdi(self, img_path, resultados_predicoes, imagem, date):
        output_data = []
        filename = img_path
        parts = filename.split('_')

        network_id = parts[0]
        leaflet_id = parts[1]
        page_number = parts[2].split('.')[0]
        page_number = page_number.split('page')[1]
        #extensao = parts[2].split('.')[1]

        data_atual = datetime.now()
        data_formatada = date


        # tratamento da detecção da oferta
        for valor in resultados_predicoes:
            annottations = []
            numero_da_oferta = 0
            for box, name in zip(valor.boxes, valor):
                name = name.names[0]
                confianca = box.conf.tolist()[0]
                coordenadas = box.xyxy.tolist()[0]
                x1_oferta, y1_oferta, x2_oferta, y2_oferta = coordenadas

                annottation = {
                    "label": name + '_' + str(numero_da_oferta), 
                    "coordinates": {
                        "x1": x1_oferta, 
                        "y1": y1_oferta,
                        "x2": x2_oferta,
                        "y2": y2_oferta 
                    },
                    "precision": confianca,
                    "description": [],
                    "price": [],
                    "dynamics": []
                }
                numero_da_oferta += 1
                annottations.append(annottation)

            informacoes_da_imagem = {
                "image": filename,
                "date": data_formatada,
                "network_id": network_id,
                "leaflet_id": leaflet_id,
                "page_number": page_number,
                "annotations": annottations
            }
            output_data.append(informacoes_da_imagem)
            

            
            for recorte in output_data[0]["annotations"]:
                coordinates = recorte["coordinates"]
                x1 = int(coordinates["x1"])
                y1 = int(coordinates["y1"])
                x2 = int(coordinates["x2"])
                y2 = int(coordinates["y2"])

                if x1 < x2 and y1 < y2:
                    imagem_recortada = imagem[y1:y2, x1:x2]

                    resultados_recorte = self.model_recorte(imagem_recortada)

                    description_list = []
                    price_list = []
                    dynamics_list = []

                    description_class = [0.0]
                    dynamics_class = [1.0]
                    price_class = [2.0]

                    oferta_detectada = False

                    for r in resultados_recorte:
                        for det, test in zip(r.boxes, r):
                            class_name = det.cls.tolist()
                            det_xywh = det.xyxy.tolist()[0]
                            x1_adj = x1 + int(det_xywh[0])
                            y1_adj = y1 + int(det_xywh[1])
                            x2_adj = x1 + int(det_xywh[2])
                            y2_adj = y1 + int(det_xywh[3])

                            if class_name == description_class:
                                description_dict = {
                                    "x1": x1_adj,
                                    "y1": y1_adj,
                                    "x2": x2_adj,
                                    "y2": y2_adj
                                }
                                description_list.append(description_dict)
                            elif class_name == price_class:
                                price_dict = {
                                    "x1": x1_adj,
                                    "y1": y1_adj,
                                    "x2": x2_adj,
                                    "y2": y2_adj
                                }
                                price_list.append(price_dict)

                            elif class_name == dynamics_class:
                                dynamics_dict = {
                                    "x1": x1_adj,
                                    "y1": y1_adj,
                                    "x2": x2_adj,
                                    "y2": y2_adj
                                }
                                dynamics_list.append(dynamics_dict)
                                oferta_detectada = True

                    if not oferta_detectada:
                        dynamics_dict = {

                        }
                        dynamics_list.append(dynamics_dict)

                    recorte["description"] = description_list
                    recorte["price"] = price_list
                    recorte["dynamics"] = dynamics_list

        return output_data

    def detect_offers(self, img_path, image, date):
        predictions = self.model_offer(image)
        json = self.json_pdi(img_path, predictions, image, date)
        return json


    # mostrar as bounding boxes caso necessário
    def draw_boxes_on_image(imagem_np, json_data):
        imagem_com_boxes = imagem_np.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2

        for informacao_da_imagem in json_data:
            for recorte in informacao_da_imagem["annotations"]:
                coordinates = recorte["coordinates"]
                x1 = int(coordinates["x"])
                y1 = int(coordinates["y"])
                x2 = int(coordinates["width"])
                y2 = int(coordinates["height"])

                # Desenha um retângulo ao redor do recorte
                cor = (0, 255, 0)  # Cor verde
                espessura = 2
                imagem_com_boxes = cv2.rectangle(imagem_com_boxes, (x1, y1), (x2, y2), cor, espessura)

                # Adiciona informações adicionais, como rótulos e preços
                for description in recorte["description"]:
                    x1_desc = int(description["x"])
                    y1_desc = int(description["y"])
                    x2_desc = int(description["width"])
                    y2_desc = int(description["height"])

                    cor = (255, 0, 0)  # Cor vermelha para descrição
                    espessura = 2
                    imagem_com_boxes = cv2.rectangle(imagem_com_boxes, (x1_desc, y1_desc), (x2_desc, y2_desc), cor, espessura)
                    label = recorte["label"]
                    cv2.putText(imagem_com_boxes, label, (x1_desc, y1_desc - 10), font, font_scale, cor, font_thickness)

                for price in recorte["price"]:
                    x1_price = int(price["x"])
                    y1_price = int(price["y"])
                    x2_price = int(price["width"])
                    y2_price = int(price["height"])

                    cor = (0, 0, 255)  # Cor azul para preço
                    espessura = 2
                    imagem_com_boxes = cv2.rectangle(imagem_com_boxes, (x1_price, y1_price), (x2_price, y2_price), cor, espessura)
                    label = recorte["label"]
                    cv2.putText(imagem_com_boxes, label, (x1_price, y1_price - 10), font, font_scale, cor, font_thickness)

        return imagem_com_boxes
