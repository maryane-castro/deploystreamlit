from PIL import Image, ImageDraw
import json

# Carregar o JSON de um arquivo
with open('final_results_pdi.json') as json_file:
    data = json.load(json_file)

# Caminho da imagem
image_path = 'noe.jpg'

# Abrir a imagem
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

# Loop através das anotações e desenhar caixas delimitadoras
for annotation in data[0]['annotations']:
    coordinates = annotation['coordinates']
    x1, y1, x2, y2 = coordinates['x1'], coordinates['y1'], coordinates['x2'], coordinates['y2']
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

# Salvar a imagem com as caixas delimitadoras desenhadas
image.save("output_image.jpg")

# Mostrar a imagem
image.show()