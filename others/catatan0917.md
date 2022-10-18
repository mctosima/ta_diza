# Cara Penggunaan Model
- `model(gambar, target)` --> model mengembalikan loss, tidak mengembalikan bounding box
- `model(gambar)` ----> model mengembalikan bounding box

# Kembalian Faster RCNN
FasterRCNN menghitung empat buah loss
- Classification Loss untuk RPN
- Regression Loss untuk RPN
- Classification Loss untuk R-CNN
- Regression loss untuk R-CNN

# Kembalian SSD
SSD menghitung dua buah loss
- Classification loss
- Regression Loss

# Contoh Keluaran
## ketika model(image,target)

```bash
LOSSES DICT ->> {'loss_classifier': tensor(0.0987, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.0919, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.0100, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0072, device='cuda:0', grad_fn=<DivBackward0>)}
```

## KELUARAN DARI VALIDASI model(images) ->>
```bash
[{'boxes': tensor([
        [39.6941, 48.9401, 79.0301, 59.4349],
        [28.9322, 50.4382, 52.1184, 56.5868],
        [46.6162, 31.9046, 55.2225, 34.6204],
        [12.9889, 23.8755, 22.9773, 36.6120]], device='cuda:0'),
        
        'labels': tensor(
        [1, 1, 1, 1],
        device='cuda:0'),
        
        'scores': tensor([
        0.8687, 0.1652, 0.1193, 0.1163, ], device='cuda:0')}]
```