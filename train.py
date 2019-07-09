"""
Train the MobileNet V2 model
"""
import os
import sys
import argparse
import pandas as pd

from mobilenet_v2 import MobileNetv2

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model


def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--classes",
        help="The number of classes of dataset.")
    # Optional arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=5,
        help="The number of train iterations.")
    parser.add_argument(
        "--weights",
        default=False,
        help="Fine tune with other weights.")
    parser.add_argument(
        "--tclasses",
        default=0,
        help="The number of classes of pre-trained model.")
    parser.add_argument(
        "--train",
        default='group0_train.txt',
        help="train file name")
    parser.add_argument(
        "--valid",
        default='group0_valid.txt',
        help="valid file name")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs), int(args.classes), int(args.size), args.weights, int(args.tclasses),str(args.train),str(args.valid))


def generate(batch, size,train_path,valid_path):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    # ptrain = 'data/train'
    # pval = 'data/validation'
    rootDir='/home/eric/data/scene'
    # train_path='group0_train.txt'
    # valid_path='group0_valid.txt'
    
    trainfilepath=os.path.join(rootDir,train_path)
    trainDf=pd.read_csv(trainfilepath) #加载papa.txt,指定它的分隔符是 \t
    trainDf.rename(columns={'image':"filename",'label':'class'},inplace=True)

    validfilepath=os.path.join(rootDir,valid_path)
    validDf=pd.read_csv(validfilepath,header=None) #加载papa.txt,指定它的分隔符是 \t
    validDf.rename(columns={0:"filename",1:'class'},inplace=True)


    # ptrain = '/home/eric/data/scene/scence#3/clutter/Normal'
    # pval = '/home/eric/data/scene/scence#3/clutter/High'


    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    # train_generator = datagen1.flow_from_directory(
    #     ptrain,
    #     target_size=(size, size),
    #     batch_size=batch,
    #     class_mode='categorical')
    train_generator = datagen1.flow_from_dataframe( dataframe=trainDf,
                                                              directory=rootDir ,
                                                              x_col="filename",
                                                              y_col="class",
                                                              subset="training",
                                                            #   classes=labels,
                                                              target_size=[size, size],
                                                              batch_size=batch,
                                                              class_mode='categorical')
    print(train_generator.class_indices)

    # validation_generator = datagen2.flow_from_directory(
    #     pval,
    #     target_size=(size, size),
    #     batch_size=batch,
    #     class_mode='categorical')
    validation_generator = datagen2.flow_from_dataframe(
                                                        dataframe=validDf,
                                                              directory=rootDir,
                                                              x_col="filename",
                                                              y_col="class",
                                                              subset="training",
                                                            #   classes=labels,
                                                              target_size=[size, size],
                                                              batch_size=batch,
                                                              class_mode='categorical')

    # count1 = 0
    # for root, dirs, files in os.walk(ptrain):
    #     for each in files:
    #         count1 += 1

    # count2 = 0
    # for root, dirs, files in os.walk(pval):
    #     for each in files:
    #         count2 += 1
    count1=trainDf.shape[0]
    count2=validDf.shape[0]

    return train_generator, validation_generator, count1, count2


def fine_tune(num_classes, weights, model):
    """Re-build model with current num_classes.

    # Arguments
        num_classes, Integer, The number of classes of dataset.
        tune, String, The pre_trained model weights.
        model, Model, The model structure.
    """
    model.load_weights(weights)

    x = model.get_layer('Dropout').output
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    model = Model(inputs=model.input, outputs=output)

    return model


def train(batch, epochs, num_classes, size, weights, tclasses,train_path,valid_path):
    """Train the model.

    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
        tclasses, Integer, The number of classes of pre-trained model.
    """

    train_generator, validation_generator, count1, count2 = generate(batch, size,train_path,valid_path)

    if weights:
        model = MobileNetv2((size, size, 3), tclasses)
        model = fine_tune(num_classes, weights, model)
    else:
        model = MobileNetv2((size, size, 3), num_classes)

    opt = Adam()
    earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=epochs,
        callbacks=[earlystop])
    rootDir='/home/eric/data/scene'
    for i in range(10):      
        test_path='group'+str(i)+'_test.txt'
        testfilepath=os.path.join(rootDir,test_path)
        testDf=pd.read_csv(testfilepath,header=None) #加载papa.txt,指定它的分隔符是 \t
        testDf.rename(columns={0:"filename",1:'class'},inplace=True)
        datagen3 = ImageDataGenerator(rescale=1. / 255)
        validation_generator = datagen3.flow_from_dataframe(
                                                            dataframe=testDf,
                                                                directory=rootDir,
                                                                x_col="filename",
                                                                y_col="class",
                                                                subset="training",
                                                                #   classes=labels,
                                                                target_size=[size, size],
                                                                batch_size=batch,
                                                                class_mode='categorical')
        
        result=model.evaluate_generator(validation_generator,steps=testDf.shape[0]//batch)
        print(result)

    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/hist.csv', encoding='utf-8', index=False)
    model.save_weights('model/weights_'+train_path.split('.')[0]+'.h5')


if __name__ == '__main__':
    main(sys.argv)
