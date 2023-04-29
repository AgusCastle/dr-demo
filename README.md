# dr-demo

# Model weights ConvNeXt y our SNF

Clone this repository and append the required folders as follows:
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

Move the models into the 'checkpoints' folder.
The images to be predicted will be stored in a path specified in ***main.py***.

Run main.py --save_path ./ --device 0