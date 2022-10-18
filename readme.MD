# Deskripsi
tba

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

# Untuk Menjalankan Program
## Proses Training

#### Perintah Dasar:

```bash
python3 src/train.py
```

#### Pilihan Argument:
```bash
usage: train.py
[-h] [--lr LR] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--model MODEL] [--pretrained PRETRAINED] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Select the learning rate (Default:5e-4)
  --batch_size BATCH_SIZE
                        Select the batch size (Default:8)
  --epochs EPOCHS       Select the number of epochs (Default:10)
  --model MODEL         Select the model (fasterrcnn, retinanet)
  --pretrained PRETRAINED
                        Select if you want to use pretrained weights
                        (Default:True)
  --seed SEED           Select the seed (Default:2022)
```

## Proses Testing

## Inferensi 1 Gambar