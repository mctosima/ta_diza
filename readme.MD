<div align="center">

# Mask Detection on Thermal Image
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>

Made available by

<a href="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"><img alt="Lightning" src="https://camo.githubusercontent.com/e8163d66137a0391b4b9e14f1f8fae8674badbcc0d26cc849815efc751be6a9c/68747470733a2f2f706c2d666c6173682d646174612e73332e616d617a6f6e6177732e636f6d2f6173736574735f6c696768746e696e672f646f63732f696d616765732f6c6f676f732f6c696768746e696e672d61692e706e67" width="200"></a>

</div>


<br>


# Instalasi

Terlebih dahulu clone repositori ini dengan cara:

```bash
git clone https://github.com/mctosima/ta_diza/tree/mct_version
```

Disarankan untuk membuat environment baru, lalu install package yang digunakan dengan cara:

**pip**

```bash
pip install -r requirements.txt
```

**conda**
```bash
conda install --file requirements.txt
```

<br>

# Dataset
Silahkan tempatkan dataset anda pada folder `data` dengan struktur folder dengan struktur sebagai berikut:

>**Warning** <br>
> Sangat disarankan untuk menggunakan platform seperti roboflow untuk mengorganisir dataset dengan format COCO


<br>

# Struktur Project
```
    .
    ├── models          # Folder untuk menyimpan model 
    ├── data            # Datset       
    │   ├── train       # Train Dataset  
    │   ├── test        # Test Dataset 
    │   └── valid       # Validation Dataset 
    │   │   ├── _annotations.coco.json    # Anotasi
    │   │   ├── images.jpg                # Gambar   
    │   │   └── ...    
    ├── others          # Histori file lama          
    ├── src             # Source code 
    └── out             # Folder untuk menyimpan hasil output
```

# Untuk Menjalankan Program
## Proses Training

#### Perintah Dasar:

```bash
python3 src/train.py
```

#### Pilihan Argument:
```bash
usage: train.py [-h] [-lr LR] [-batch_size BATCH_SIZE] [-epochs EPOCHS]
                [-model MODEL] [-pretrained] [-seed SEED] [-runname RUNNAME]
                [-patience PATIENCE] [-opt OPT] [-sched SCHED]

To do the training process with some options

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                Select the learning rate (Default:5e-4)
  -batch_size BATCH_SIZE
                        Select the batch size (Default:8)
  -epochs EPOCHS        Select the number of epochs (Default:10)
  -model MODEL          Select the model (fasterrcnn, retinanet)
  -pretrained           Select if you want to use pretrained weights
                        (Default:True)
  -seed SEED            Define the seed (Default:2022)
  -runname RUNNAME      Define the run name
  -patience PATIENCE    Define the patience for early stopping
  -opt OPT              Define the optimizer (sgd, adam, rmsprop, adagrad,
                        adamw), Default: sgd
  -sched SCHED          Define the scheduler (cosine, step, linear,
                        exponential, plateau). Default: None
```

## Proses Testing

#### Perintah Dasar:


```bash
python3 src/test.py
```

#### Pilihan Argument:
```bash
usage: test.py [-h] [-model MODEL] [-pth PTH] [-batch_size BATCH_SIZE]
               [-exportres]

Testing Mask Detection Model

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL          Model to use for testing
  -pth PTH              Name of saved pytorch model
  -batch_size BATCH_SIZE
                        Batch size for testing
  -exportres            Export the result to txt file
```


## Inferensi 1 Gambar

#### Perintah Dasar:

```bash
python3 src/inference.py
```

#### Pilihan Argument:
```bash
usage: inference.py [-h] [-plotshow] [-passimage] [-model MODEL] [-pth PTH]
                    [-img IMG]

Inference Mode

optional arguments:
  -h, --help    show this help message and exit
  -plotshow     Show plot or save plot to output folder
  -passimage    Use test dataset or define the image path
  -model MODEL  Model to use for inference
  -pth PTH      Name of saved pytorch model
  -img IMG      Name of image to use for inference
```
<br>

# Inferensi Dengan ONNX
>**Note** <br>
>Anda disarankan menggunakan model inferensi ini apabila anda hanya ingin melakukan deteksi masker tanpa harus menggunakan/menginstall pytorch atau pytorch lightning

TBA

<br>

# Link Download
### Wandb Report
[Wandb Link](https://wandb.ai/mctosima/diza-mask-detection-thermal)

### Download Model (Google Drive)
[Link](https://drive.google.com/drive/folders/1y1BiRtbh15HtIkb9jYl-NYY9QGu7pE4l?usp=sharing)