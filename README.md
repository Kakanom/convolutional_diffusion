# An analytic theory of creativity in convolutional diffusion models

Code for the paper "An Analytic Theory of Creativity" (https://arxiv.org/abs/2412.20292)

Use the following command to train diffusion models:


```
python training_script.py --epochs 300 --dataset cifar10 --conditional --mode zeros --layers 8 --resnet --homedir [directory for model checkpoints]
```

This specific command will train an 8-layer class-conditional ResNet on cifar10 without normalization for 300 epochs, using default lr, batchsize, and lr decay settings.

Use the following command to calibrate scales for a trained diffusion model:


```
python scales_calibration.py --dataset cifar10 --modelfile [model file name] --kfilename scalesfile --tld [directory for scales files] --scoremoduletype [one of LS, ELS, bbELS] --conditional --kernelsizes 3 5 7 9 11 13 15 17 --nsteps 20 --nsamps 10
```

This specific command will calibrate a particular class-conditional score machine (LS, ELS, or boundary-broken ELS) on cifar10 and save the scales file to a file at scalesfile.pt, checking scales across the entire range 3…17 for a 20-step reverse process, taking the median across 10 samples.

Use the following command to generate a directory with (E)LS samples:


```
python els_script.py --expname cifar10_resnet_els --dataset cifar10 --scoremoduletype [one of LS, ELS, bbELS] --conditional --scalesfile [path to scales] --numiters 100
```

This specific command will generate a directory cifar10_resnet_els with 100 (E)LS outputs and corresponding input seeds.

`utils/patch_knn.py` - реализация kNN на PyTorch + версия на KD-tree

### Запуск kNN
```bash
python knn_demo.py
```

### Запуск kNN с отображением картинок
```bash
python knn_demo_show.py
```
