import tensorflow as tf

#Mask detection sysytem(Yes/No)--->Classification
#Object detection system
     #Collect images of all objects
     #Make test and train folder
     #Make folders of each class images in both test and train
     #Make image data gererator
     #Make CNN
     #Training

#Data preprocessing means make over data simple as possible and make over data in such a 
#way machine can learn easily     
#Data Augmentation-->Is concept of making more data from previous data     

#image data generator will make the batches of images by augmenting them
train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2,
                             rotation_range=0.2,shear_range=0.2,rescale=1/255)

#rotation_range->int,float,degree range for random rotations
#zoom_range->Float, range for random zoom
#shear_range->Float,Shear_intensity(Shear angle in counter-clockwise directions in degree)
#In rescale we provide a value to which pixels are multiplied

test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)


#Providing path of images to both generator
train_dataset=train_datagen.flow_from_directory("train",target_size=(150,150),
                    class_mode='binary',batch_size=16)

#target_size->It is the size of your input image, every image will be resized to
#this size
#class_mode->Set 'binary' if you have only two classes to predict,if not set to 'categorical'
#batch_size->No of images to be provided from the generator per batch


test_dataset=test_datagen.flow_from_directory("test",target_size=(150,150),
                    class_mode='binary',batch_size=16)


'''
Building CNN model
'''

cnn=tf.keras.models.Sequential()


#Kernal--->It is a nan matrix(filter) which is convolved over images and extract 
#particular feature from that image
#Kernal results in feature map

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,
                               input_shape=(150,150,3),activation='relu'))

#filters->Dimensionality of the output space
#kernal_size->An integer,specifying the hieght and width of the 2D convolution window
#activation->If you don't specify anything, no activation is applied

#When kernal size is small than feature map will contain smaller information
#also which also result in slow speed

#when kernal size is big then speed will fast but accuracy can be low

#Pooling-->Reducing the size of feature map by taking average, max of each
#window in feature map

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#pool_size->specifying dimension of window
#strides->specifies how far the pooling window moves for each pooling step,if none it will
#default to pool size

#We can also make features from previous feature
#adding next cnn layer
cnn.add(tf.keras.layers.Conv2D(filters=16,kernel_size=3,activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#flatten layers is responsible to convert data into 1D so that to fed up hidden layers
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=121,activation='relu'))#regression

cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))#softmax(classification)

'''
Compiling
'''
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')

'''
fit
'''
cnn.fit(train_dataset,validation_data=test_dataset,epochs=50)

#At last tell validation loss

cnn.save('Harsh.h5')

#Sigmoid

#Note:- When third class is possible use softmax


#Image--->150,150,3
#original shape while feeding to cnn-->16,150,150,3
#Model always assume first value as batch value