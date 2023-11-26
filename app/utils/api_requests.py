import requests
import json
import numpy as np
from PIL import Image
from io import BytesIO


email = "ifce@ifce.gob.br"
password = "ifce"


def get_access_token():
    """
    Computa o token para o acesso à API.

    Returns:
        str: token de acesso.
    """
    raw_data = '{"email":"' + email + '","password":"' + password + '"}'
    headers = {'Content-Type':'application/json'}
    url = "http://app.infomarketpesquisa.com/api/users/login"
    response = requests.post(url, data=raw_data, headers=headers)
    data = json.loads(response.text)
    return data["id"]


token = get_access_token()


def get_leaflet_page_crops(leaflet_id, page):
    """
    Realiza o recorte de uma imagem.

    Args:
        image (ndarray): imagem.
        x1 (float): coordenada x mínima.
        y1 (float): coordenada y mínima.
        x2 (float): coordenada x máxima.
        y2 (float): coordenada y máxima.

    Returns:
        ndarray: imagem recortada.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': token
    }
    url = f"http://app.infomarketpesquisa.com/api/leafletPageCrops/filterByPage?leafletId={leaflet_id}&page={page}"

    response = requests.get(url, headers=headers)
    data = json.loads(response.text)

    return data


def get_leaflet_image(date, leaflet_id, page_number):
    """
    Recupera um determinado encarte da API.

    Args:
        date (str): data do encarte no formato YYYY-M.
        leaflet_id (str): id do encarte.
        page_number (str): número da página do encarte.

    Returns:
        ndarray: imagem do encarte.
    """
    url = f"https://s3.amazonaws.com/encartes/{date}/{leaflet_id}-{page_number}.jpg"
    try:
        response = requests.get(url)
        response.raise_for_status()
        leaflet_image = np.array(Image.open(BytesIO(response.content)))
        return leaflet_image
    except:
        raise Exception(f'Url not found: {url}')


def get_leaflet_image_local():
    pass