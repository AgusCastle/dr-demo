# dr-demo

# Pesos de los modelos ConvNeXt y nuestra SNF

Clona este repositorio y agrega las carpetas que falten como se muestra:
````
dr-demo
    |
    |--checkpoints
    |           |-- convnext_0000.pth
    |           |-- convnext_###1.pth
    |           |-- .................
    |           |-- snfw2.pt
    |
    |--images
    |      |-- img_0.jpg
    |      |-- img_n.*
    |
    |-- models
    |-- utils
    |-- demo.py
    |-- main.py
    |-- readme.md
````

Download models [here](https://drive.google.com/drive/folders/18FrgUq9dw8Ww0Z0brVnBBz3TduTZGb_n?usp=sharing)

Mover los modelos dentro de la carpeta 'checkpoints'.
Las imagenes que se predeciran seran guardas en una ruta especificada en main.py

Run main.py --save_path ./ --device 0