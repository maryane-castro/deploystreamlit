
<h1 align='center'> 💿 Instalação</h1>
Siga as etapas abaixo para configurar o ambiente de desenvolvimento:

1. Clone este repositório para o seu sistema local.

2. Baixe o [modelo NER](https://drive.google.com/file/d/121Q3hQRu4bNQH7zKoE6GLQ00jdQfBuXj/view?usp=sharing) e o descompacte no diretório `bd_utils`.

3. Baixe o [modelo RECORTE](https://www.dropbox.com/scl/fi/720wak5rw69cxenhp2793/recortes.pt?rlkey=6063n0m1xs2xtqp7zp7l4dwgo&dl=0) e coloque-o no `diretório models/pdi`

4. Utilize o seguinte comando para criar uma ambiente com o `conda`:

    ```bash
    conda create -n sale -c conda-forge python=3.9 cudatoolkit=11.7 cudnn=8.4.1
    ```

    Isso irá criar um ambiente com o Python 3.9 e as dependências do Cuda.

5. Instale no ambiente criado as dependências do projeto utilizando o `pip`:

    ```bash
    pip install -r requirements.txt
    ```
    