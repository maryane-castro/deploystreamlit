import numpy as np


def cut_image(image, x1, y1, x2, y2):
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
    return image[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
