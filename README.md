### Mini-project summary
#### Goal
Make a series of experiments (mini project) in the field of Computer Vision & Deep Learning
#### Data
[cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) from keras.datasets
#### Software
Keras, Tensorflow
#### Model
Initial architecture: AlexNet like:
* https://github.com/arxitekton/CV/blob/master/notebooks/cifar10_alexnet_mod.ipynb
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 64)        1792      
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 64)        256       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 64)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 128)       401536    
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 128)       512       
_________________________________________________________________
activation_2 (Activation)    (None, 16, 16, 128)       0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 128)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 192)         221376    
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 192)         768       
_________________________________________________________________
activation_3 (Activation)    (None, 8, 8, 192)         0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 192)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 4, 256)         442624    
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 256)         1024      
_________________________________________________________________
activation_4 (Activation)    (None, 4, 4, 256)         0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 4, 4, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
batch_normalization_5 (Batch (None, 4096)              16384     
_________________________________________________________________
activation_5 (Activation)    (None, 4096)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
batch_normalization_6 (Batch (None, 4096)              16384     
_________________________________________________________________
activation_6 (Activation)    (None, 4096)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                40970     
_________________________________________________________________
batch_normalization_7 (Batch (None, 10)                40        
_________________________________________________________________
activation_7 (Activation)    (None, 10)                0         
=================================================================
Total params: 34,706,290
Trainable params: 34,688,606
Non-trainable params: 17,684
```
Model training history:
![model loss](https://github.com/arxitekton/CV/blob/master/results/img/alexnet_like_loss.png)
![model accuracy](https://github.com/arxitekton/CV/blob/master/results/img/alexnet_like_accuracy.png)

Used metric: accuracy
achieved accuracy: 0.7191 (60 epochs)

I used accuracy only because it is most common, most straightforward evaluation criteria (don’t have
much time). But it is very easy to understand and interpret this metric. Nevertheless, when the class
distribution is unbalanced, the metric becomes useless :(.
There is a lot another one metrics, like: Confusion Matrix, precision, recall, Sensitivity/Specificity, f1
score, ROC/AUC...

Improved architecture:
* https://github.com/arxitekton/CV/blob/master/notebooks/cifar10_alexnet_mod.ipynb
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_11 (Conv2D)           (None, 32, 32, 32)        896       
_________________________________________________________________
activation_14 (Activation)   (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_14 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 32, 32, 32)        9248      
_________________________________________________________________
activation_15 (Activation)   (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_15 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 16, 16, 64)        18496     
_________________________________________________________________
activation_16 (Activation)   (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_16 (Batc (None, 16, 16, 64)        256       
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 16, 16, 64)        36928     
_________________________________________________________________
activation_17 (Activation)   (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_17 (Batc (None, 16, 16, 64)        256       
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 8, 8, 128)         73856     
_________________________________________________________________
activation_18 (Activation)   (None, 8, 8, 128)         0         
_________________________________________________________________
batch_normalization_18 (Batc (None, 8, 8, 128)         512       
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 8, 8, 128)         147584    
_________________________________________________________________
activation_19 (Activation)   (None, 8, 8, 128)         0         
_________________________________________________________________
batch_normalization_19 (Batc (None, 8, 8, 128)         512       
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 4, 4, 128)         0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 4, 4, 128)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 10)                20490     
=================================================================
Total params: 309,290
Trainable params: 308,394
Non-trainable params: 896
```
Model training history:
![model loss](https://github.com/arxitekton/CV/blob/master/results/img/mod_wo_aug_adam_loss.png)
![model accuracy](https://github.com/arxitekton/CV/blob/master/results/img/alexnet_like_accuracy.png)

achieved accuracy: 0.8559 (60 epochs)

#### Experiments
##### With real-time data augmentation:
```
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

```
Model training history:
![model loss](https://github.com/arxitekton/CV/blob/master/results/img/mod_aug_adam_loss.png)
![model accuracy](https://github.com/arxitekton/CV/blob/master/results/img/mod_aug_adam_accuracy.png)

achieved accuracy: 0.8712 (60 epochs)

##### Experiments with optimizers:
* https://github.com/arxitekton/CV/blob/master/notebooks/cifar10_alexnet_mod.ipynb

optimizer | accuracy
--------- | --------
aug_adagrad|0.8996
aug_adamax|0.887
aug_adam|0.8712
aug_rmsprop|0.8676
wo_aug_adam|0.8559
alexnet_like|0.7191

Model training history with adagrad:
![model loss](https://github.com/arxitekton/CV/blob/master/results/img/mod_aug_adagrad_loss.png)
![model accuracy](https://github.com/arxitekton/CV/blob/master/results/img/mod_aug_adagrad_accuracy.png)


##### Experiments with Learning rate schedulers:
* https://github.com/arxitekton/CV/blob/master/notebooks/cifar10_alexnet_mod-adagrad_lrsscheduler_step_decay.ipynb
* https://github.com/arxitekton/CV/blob/master/notebooks/cifar10_alexnet_mod-adagrad_lrsscheduler_exp_decay.ipynb
```
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

```
```
    def exp_decay(epoch):
        initial_lrate = 0.1
        k = 0.1
        lrate = initial_lrate * math.exp(-k*epoch)
        return lrate

```
```
    def custom_exp_decay(epoch):
        initial_lrate = 0.01
        if epoch > 60:
            k = 0.05
            lrate = initial_lrate * math.exp(-k*epoch)
            return lrate
        else:
            return initial_lrate

```
Model training history with step_decay:
![model loss](https://github.com/arxitekton/CV/blob/master/results/img/step_decay_loss.png)
![model accuracy](https://github.com/arxitekton/CV/blob/master/results/img/step_decay_accuracy.png)

To improve accuracy, we can increase the number of epochs, play with augmentation, with optimizers and their parameters.
maybe revise the architecture

