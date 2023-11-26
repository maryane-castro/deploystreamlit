
import cv2
from ultralytics import YOLO
from datetime import datetime
import json


modelo_oferta = YOLO('models/pdi/ofertas.pt')
modelo_recorte = YOLO('models/pdi/recortes.pt')


def pipelinePDI(image_file, image_path, modelo_oferta=modelo_oferta, modelo_recorte=modelo_recorte):

    parts = image_file.split('_')
    network_id = parts[0]
    leaflet_id = parts[1]
    page_number = parts[1].split('.')[0]

    data_atual = datetime.now()
    data_formatada = data_atual.strftime('%Y-%m-%d')

    # Time Start
    results = modelo_oferta(image_path)

    output_data = []

    for r in results:
        annotations = []

        for det, test in zip(r.boxes, r):
            name = test.names[0]
            conf = det.conf.tolist()[0]
            det_xywh = det.xyxy.tolist()[0]
            x, y, w, h = det_xywh

            annotation = {
                "label": name,
                "coordinates": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "precision": conf,
                "description": [],
                "price": [],
                "dynamics": []
            }
            annotations.append(annotation)

        image_info = {
            "image": image_path,
            "date": data_formatada,
            "network_id": network_id,
            "leaflet_id": leaflet_id,
            "page_number": page_number,
            "annotations": annotations
        }

        output_data.append(image_info)

    imagem_original = cv2.imread(image_path)

    if imagem_original is None or imagem_original.shape[0] <= 0 or imagem_original.shape[1] <= 0:
        print(f"A imagem {image_path} possui dimensões inválidas e não será processada.")
       

    for recorte in output_data[0]["annotations"]:
        roi = (
            recorte["coordinates"]["x"],
            recorte["coordinates"]["y"],
            recorte["coordinates"]["width"],
            recorte["coordinates"]["height"]
        )

        x1, y1, x2, y2 = map(int, roi)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(imagem_original.shape[1], x2)
        y2 = min(imagem_original.shape[0], y2)

        if x1 < x2 and y1 < y2:
            imagem_recortada = imagem_original[y1:y2, x1:x2]

            resultados = modelo_recorte(imagem_recortada)

            description_list = []
            price_list = []
            dynamics_list = []

            description_class = [0.0]
            dynamics_class = [1.0]
            price_class = [2.0]

            oferta_detectada = False

            for r in resultados:
                for det, test in zip(r.boxes, r):
                    class_name = det.cls.tolist()
                    det_xywh = det.xyxy.tolist()[0]
                    x1_adj = x1 + int(det_xywh[0])
                    y1_adj = y1 + int(det_xywh[1])
                    x2_adj = x1 + int(det_xywh[2])
                    y2_adj = y1 + int(det_xywh[3])

                    if class_name == description_class:
                        description_dict = {
                            "x": x1_adj,
                            "y": y1_adj,
                            "width": x2_adj,
                            "height": y2_adj
                        }
                        description_list.append(description_dict)
                        cv2.rectangle(imagem_original, (x1_adj, y1_adj), (x2_adj, y2_adj), (0, 255, 0), 2)
                        cv2.putText(imagem_original, "Description", (x1_adj, y1_adj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif class_name == price_class:
                        price_dict = {
                            "x": x1_adj,
                            "y": y1_adj,
                            "width": x2_adj,
                            "height": y2_adj
                        }
                        price_list.append(price_dict)
                        cv2.rectangle(imagem_original, (x1_adj, y1_adj), (x2_adj, y2_adj), (0, 255, 0), 2)
                        cv2.putText(imagem_original, "Price", (x1_adj, y1_adj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif class_name == dynamics_class:
                        dynamics_dict = {
                            "x": x1_adj,
                            "y": y1_adj,
                            "width": x2_adj,
                            "height": y2_adj
                        }
                        dynamics_list.append(dynamics_dict)
                        cv2.rectangle(imagem_original, (x1_adj, y1_adj), (x2_adj, y2_adj), (0, 255, 0), 2)
                        cv2.putText(imagem_original, "Dynamics", (x1_adj, y1_adj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        oferta_detectada = True

            if not oferta_detectada:
                dynamics_dict = {
                    "padrao": "oferta_de_por"
                }
                dynamics_list.append(dynamics_dict)

            recorte["description"] = description_list
            recorte["price"] = price_list
            recorte["dynamics"] = dynamics_list

            cv2.rectangle(imagem_original, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caixa delimitadora

    return output_data


teste = pipelinePDI('idrede_ideencarte_page1.jpg', 'idrede_ideencarte_page1.jpg')

print(teste)