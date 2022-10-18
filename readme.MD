# Deskripsi
tba

<br>


# Instalasi
## Menggunakan pip

Terlebih dahulu clone repositori ini dengan cara:

```bash
git clone xxx
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

# Hasil (Wandb)
[Wandb Link](https://wandb.ai/mctosima/diza-mask-detection-thermal)