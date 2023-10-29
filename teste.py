import gdown
import os

# ID do arquivo no Google Drive
file_id = '1xyHv-3Nt-hfn_sMBxuc4k3GPl1ivA4nH'

# URL de download do arquivo
url = f'https://drive.google.com/uc?id={file_id}'

# Pasta de destino onde você deseja salvar o arquivo
destination = 'models/pdi/'

# Verifique se a pasta de destino existe, caso contrário, crie-a
os.makedirs(os.path.dirname(destination), exist_ok=True)

# Faça o download do arquivo do Google Drive
gdown.download(url, destination, quiet=False)

# Agora o arquivo está na pasta 'models/pdi' e você pode executar o seu código
