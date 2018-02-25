from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge,Flatten
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Lambda, K, LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.backend import shape
import scipy.io


import time
import numpy as np

np.random.seed(1337)


class CustomImageDataGenerator(ImageDataGenerator):
    """
    Because Xception utilizes a custom preprocessing method, the only way to utilize this
    preprocessing method using the ImageDataGenerator is to overload the standardize method.

    The standardize method gets applied to each batch before ImageDataGenerator yields that batch.
    """

    def standardize(self, x):
        """
        Taken from keras.applications.xception.preprocess_input
        """
        if self.featurewise_center:
            x /= 255.
            x -= 0.5
            x *= 2.
        return x
#----------mat file----------------
train_data = []
train_label = []
val_data = []
val_label = []
mat_contents_train = scipy.io.loadmat('images/train.mat')
train = mat_contents_train ['images']
for x in range (0,400):
    if x %2 ==0 :
        train_data.append(train[0][x])
    else :
        train_label.append(train[0][x])

mat_contents_val = scipy.io.loadmat('images/val.mat')
val = mat_contents_val ['images']
for x in range (0,240):
    if x %2 ==0 :
        val_data.append(val[0][x])
    else :
        val_label.append(val[0][x])

#----------------------------------



def get_training_generator(batch_size=16):
    train_data_dir = 'images/train/'
    validation_data_dir = 'images/val/'
    image_datagen = CustomImageDataGenerator(featurewise_center=True)

    train_generator = image_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size
    )

    val_generator = image_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator


def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image"""
    return K.mean(input, axis=3)


def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]


batch_size_phase_one = 16
batch_size_phase_two = 16
nb_val_samples = 120

nb_epochs = 10

img_width = 299
img_height = 299

# Setting tensorbord callback
now = time.strftime("%c")
tensorboard_callback = TensorBoard(log_dir='logs/' + 'cnn_rnn ' , histogram_freq=0, write_graph=True,
                                   write_images=False)

# Loading dataset
print("Loading the dataset with batch size of {}...".format(batch_size_phase_one))
train_generator, val_generator = get_training_generator(batch_size_phase_one)
print("Dataset loaded")

print("Building model...")
input_tensor = Input(shape=(img_width, img_height, 3))

# Creating CNN
cnn_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = cnn_model.output
cnn_bottleneck = Flatten()(x)#flatten layer
#x= Flatten()(x)

# Make CNN layers not trainable
for layer in cnn_model.layers:
    layer.trainable = False

# Creating RNN
x = Lambda(rgb_to_grayscale, rgb_to_grayscale_output_shape)(input_tensor)
x = Reshape((23, 3887))(x)  # 23 timesteps, input dim of each timestep 3887
x = LSTM(41472, return_sequences=True)(x)
rnn_output = LSTM(41472)(x)

# Merging both cnn bottleneck and rnn's output wise element wise multiplication
x = merge([cnn_bottleneck, rnn_output], mode='mul') #falten hna
x= Flatten()(x)
x = Dense(11776, activation='relu')(x)
#layer 3ded al nerons feha 
predictions = Dense(4, activation='softmax')(x)

model = Model(input=input_tensor, output=predictions)

print("Model built")

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')#adam optimzer
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])
model.summary()
print("Starting training")
checkpointer = ModelCheckpoint(filepath="initial_cnn_rnn_weights_2.hdf5", verbose=1, save_best_only=True)
#model.fit_generator(train_generator, samples_per_epoch=100, nb_epoch=nb_epochs, verbose=1,
#                    validation_data=val_generator,
#                    nb_val_samples=nb_val_samples,
#                    callbacks=[tensorboard_callback, checkpointer])

print("Initial training done, starting phase two (finetuning)")
#Load two new generator with smaller batch size, needed because using the same batch size
#for the fine tuning will result in GPU running out of memory and tensorflow raising an error
print("Loading the dataset with batch size of {}...".format(batch_size_phase_two))
train_generator, val_generator = get_training_generator(batch_size_phase_two)
print("Dataset loaded")

# Load best weights from initial training
model.load_weights("initial_cnn_rnn_weights_2.hdf5")

# Make all layers trainable for finetuning
set_trainable=False
for layer in model.layers:    
    layer.trainable = True
#    if layer.name == 'block5_conv1': 
#        set_trainable = True 
#    if set_trainable: 
#        layer.trainable = True 
#    else: layer.trainable = False


model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

checkpointer = ModelCheckpoint(filepath="finetuned_cnn_rnn_weights_2.hdf5", verbose=1, save_best_only=True,
                               monitor='val_acc')
model.fit_generator(train_generator, samples_per_epoch=100, nb_epoch=nb_epochs, verbose=1,
                    validation_data=val_generator,
                    nb_val_samples=nb_val_samples,
                    callbacks=[tensorboard_callback, checkpointer])

# Final evaluation of the model
print("Training done, doing final evaluation...")

model.load_weights("./finetuned_cnn_rnn_weights_2.hdf5")

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

scores = model.evaluate_generator(val_generator, val_samples=nb_val_samples)
print(model.metrics_names, scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
