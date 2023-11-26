import json


def read_pdi_json(path_json):
    """
    Realiza a leitura do JSON de saída do PDI e retorna seus dados.

    Args:
        path_json (str): caminho para o JSON de saída do PDI.

    Returns:
        str: id do encarte.
        str: data do encarte no formato YYYY-M.
        str: número da página do encarte.
        list: lista contendo as informações das ofertas detectadas no encarte pelo PDI.
    """
    with open(path_json) as f:
        json_content = json.load(f)[0]
        leaflet_id = json_content['leaflet_id']
        date = json_content['date']
        page_number = json_content['page_number']
        offers = json_content['annotations']
    return leaflet_id, date, page_number, offers


def get_intermediate_json_template():
    """
    Faz a leitura do template do JSON intermediário.

    Returns:
        dict: dicionário contendo as informações do template do JSON intermediário.
    """
    with open('utils/intermediate_json_template.json') as f:
        json_content = json.load(f)
    return json_content


def fill_intermediate_json(json_dict, leaflet_id, page_number, offer, ocr_result):
    """
    Preenche o JSON intermediário com as informações de uma oferta.

    Args:
        json_dict (dict): dicionário contendo o JSON intermediário.
        leaflet_id (str): id do encarte.
        page_number (str): número da página do encarte.
        offer (dict): dicionário contendo as informações de uma oferta computadas pelo PDI
        ocr_result (dict): dicionário contendo as informações de uma oferta computadas pelo OCR

    Returns:
        dict: dicionário do JSON intermediário preenchido com os resultados obtidos da oferta.
    """
    json_dict['pageCrop']['leafletId'] = leaflet_id
    json_dict['pageCrop']['page'] = page_number
    json_dict['pageCrop']['x1'] = offer['coordinates']['x1']
    json_dict['pageCrop']['y1'] = offer['coordinates']['y1']
    json_dict['pageCrop']['x2'] = offer['coordinates']['x2']
    json_dict['pageCrop']['y2'] = offer['coordinates']['y2']
    json_dict['pageCrop']['PageCropItems'][0]['itemId'] = offer['label']
    for price, dynamic in zip(ocr_result['prices'], ocr_result['dynamics']):
        json_dict['pageCrop']['PageCropItems'][0]['pageCropItemPrices'].append({
            'value': price,
            'dynamic': dynamic,
            'minimumQuantity': 1,
            'mse': 1
        })
    json_dict['pageCrop']['PageCropItems'][0]['similarity'] = 0.7  # ???
    for j in range(len(ocr_result['descriptions'])):
        x1 = offer['description'][j]['x1']
        y1 = offer['description'][j]['y1']
        x2 = offer['description'][j]['x2']
        y2 = offer['description'][j]['y2']
        json_dict['pageCrop']['ocrJson']['textAnnotations'].append({
            'description': ocr_result['descriptions'][j],
            'boundingPoly': {'vertices': [
                {'x': x1, 'y': y1},
                {'x': x1, 'y': y2},
                {'x': x2, 'y': y1},
                {'x': x2, 'y': y2}
            ]}
        })
    for j in range(len(ocr_result['prices'])):
        x1 = offer['price'][j]['x1']
        y1 = offer['price'][j]['y1']
        x2 = offer['price'][j]['x2']
        y2 = offer['price'][j]['y2']
        json_dict['pageCrop']['ocrJson']['textAnnotations'].append({
            'description': ocr_result['prices'][j],
            'boundingPoly': {'vertices': [
                {'x': x1, 'y': y1},
                {'x': x1, 'y': y2},
                {'x': x2, 'y': y1},
                {'x': x2, 'y': y2}
            ]}
        })
    return json_dict


def save_nlp_intermediate_json(json_path, leaflet_id, page_number, offers, ocr_results):
    """
    Salva o JSON intermediário de saída do NLP.

    Args:
        json_path (str): caminho para salvar o JSON intermediário de saída do NLP.
        leaflet_id (str): id do encarte.
        page_number (str): número da página do encarte.
        offers (list): lista contendo as informações das ofertas detectadas no encarte pelo PDI.
        ocr_results (list): lista contendo as informações das ofertas computadas pelo OCR.

    Returns:
        None
    """
    json_data = []
    for offer, ocr_result in zip(offers, ocr_results):
        json_dict = get_intermediate_json_template()
        json_dict = fill_intermediate_json(json_dict, leaflet_id, page_number, offer, ocr_result)
        json_data.append(json_dict)

    json_object = json.dumps(json_data, indent=4)
    with open(json_path, 'w') as outfile:
        outfile.write(json_object)


def save_nlp_final_json(json_path, bd_results, leaflet_id, page_number, offers, similarity_threshold=0.7):
    """
    Salva o JSON final de saída do NLP.

    Args:
        json_path (str): caminho para salvar o JSON final de saída do NLP.
        bd_results (list): lista contendo o resultado da busca no banco das ofertas realizado pelo NLP.
        leaflet_id (str): id do encarte.
        page_number (str): número da página do encarte.
        offers (list): lista contendo as informações das ofertas detectadas no encarte pelo PDI.
        similarity_threshold (float): limiar de similaridade dos itens pesquisados no banco (default=0.7).

    Returns:
        None
    """
    json_data = {'leaflet_id': leaflet_id, 'page': page_number, 'leaflet_items': [], 'unidentified_leaflet_items_ids': []}
    for offer_result, offer in zip(bd_results, offers):
        for product_result in offer_result:
            if product_result['similarity'] >= similarity_threshold:
                json_data['leaflet_items'].append({
                    'item_id': offer['label'],
                    'product_id': product_result['product_id'],
                    'value': product_result['price'],
                    'dynamic': product_result['dynamic'],
                    'minimum_quantity': 1
                })
            elif offer['label'] not in json_data['unidentified_leaflet_items_ids']:
                json_data['unidentified_leaflet_items_ids'].append(offer['label'])

    json_object = json.dumps(json_data, indent=4)


    with open(json_path, 'w') as outfile:
        outfile.write(json_object)

    return json_object