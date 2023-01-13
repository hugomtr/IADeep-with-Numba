import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense, MaxPooling1D, Conv1D, Flatten, BatchNormalization, Activation, GlobalAveragePooling1D, Dropout, ReLU, Concatenate, Input, add
from keras import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os 
import argparse

Y = np.load('data/Y.npy')
X = np.load('data/X.npy')

N_RESNET_BLOCKS = 3
N_LEVELS = 1
N_EPOCHS = 50
BATCH_SIZE = 50

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=bool, default=False, help="Launch the training of the model")
parser.add_argument("--res_blocks", type=int, default=N_RESNET_BLOCKS, help="Num of res block per level")
parser.add_argument("--levels", type=int, default=N_LEVELS, help="different levels for the network must be > 0, level = i means the network will use conv layers wth increasing num of filters = 2**k for k in range(1,i)")
parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
parser.add_argument("--dropout", type=bool, default=False, help="Adding dropout")
args = parser.parse_args()

def resnet_block(input_data,filters,conv_size):
  x = Conv1D(filters, conv_size, padding='same', activation='relu')(input_data)
  x = BatchNormalization()(x)
  x = Conv1D(filters, conv_size, padding='same', activation='relu')(x)
  x = BatchNormalization()(x)
  x = add([x, input_data])
  x = Activation('relu')(x)
  return x

def CustomResNet():
  inputs = Input(shape=(3, 64))

  # First convolutional layer
  x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
  x = MaxPooling1D(2, strides=None, padding='same')(x)

  filters = 64   
  for i in range(args.levels):
    for j in range(args.res_blocks):
      conv_size = 3 if j > 1 else 1
      x = resnet_block(x, filters, conv_size)
    filters *= 2
    x = Conv1D(filters, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, strides=None, padding='same')(x)

  x = GlobalAveragePooling1D()(x)
  x = Dense(64, activation='softmax')(x)
  if args.dropout:
    x = Dropout(0.2)(x)
  return Model(inputs,x)


model = CustomResNet()

def train_model():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42,shuffle = True)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42,shuffle = True)

    model.compile(loss = keras.losses.CategoricalCrossentropy(from_logits=False),
                    optimizer = 'SGD',
                    metrics   = 'accuracy')

    ### Training
    history  = model.fit(X_train, Y_train, verbose=1, batch_size=args.batch_size, epochs=args.epochs,validation_data=(X_val, Y_val))

    ### Saving figures
    plt.plot( history.history["loss"], "b",label="train loss")
    plt.plot( history.history["val_loss"], "r",label="val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.isdir("output"):
        try:
            os.mkdir("output")
        except OSError as error:
            print(error)
    plt.savefig( "output/loss.png")


    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(X_test, Y_test)
    print("test loss, test acc:", results)


    ### Saving models
    if not os.path.isdir("weights"):
        try:
            os.mkdir("weights")
        except OSError as error:
            print(error)
            exit()

    model.save_weights("weights/weights052")

if (args.train):    
    train_model()
