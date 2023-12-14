from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (5, 5), input_shape = (240,240, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.25))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.25))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('C:\\Users\\Puneeths\\Desktop\\Final project\\datasets\\train',
target_size = (240,240),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('C:\\Users\\Puneeths\\Desktop\\Final project\\datasets\\validation',
target_size = (240,240),
batch_size = 32,
class_mode = 'binary')


batch_size=32
model_info=classifier.fit_generator(training_set,
steps_per_epoch = 100/batch_size,
epochs = 30,
validation_data = test_set,
validation_steps = 30/batch_size)


### Performance evaluation
#########################
score = classifier.evaluate_generator(test_set,155/batch_size)
print(" Total: ", len(test_set.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])
#print("Accuracy = ",score[1])

classifier.save('C:\\Users\\Puneeths\\Desktop\\Final project\\glaucoma.h5')


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt


target_size = (240,240)
model=load_model('C:\\Users\\Puneeths\\Desktop\\Final project\\glaucoma.h5')
print("model loaded")

for i in range(1,66):
    test_image = image.load_img('C:\\Users\\Puneeths\\Desktop\\Final project\\datasets\\validation\\glau\\1 ('+str(i)+').png', target_size = (240,240))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
          print('glaucoma')
          cv2.imwrite('C:\\Users\\Puneeths\\Desktop\\pg1'+str(i)+'.png',test_image)

#         account_sid = 'AC705e3ede63faa2ce218a58fcbe4465c7'
#         auth_token = '6ae316781c8d00b79f4cc1f792c9ae10'
#         client = Client(account_sid, auth_token)
#
#         message = client.messages \
#                          .create(
#                             body="glaucoma",
#                             from_='+12054028214',
#                             to='+919108545745'
#                           )
#
#         print(message.sid)
 
import numpy as np
from keras.preprocessing import image
for i in range(1,91):
    test_image = image.load_img('C:\\Users\\Puneeths\\Desktop\\Final project\\datasets\\test\\normal\\1 ('+str(i)+').png', target_size = (240,240))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    training_set.class_indices
    if result[0][0] != 1:
        print('normal')
        cv2.imwrite('C:\\Users\\Puneeths\\Desktop\\pg1'+str(i)+'.png',test_image)

#       account_sid = 'AC705e3ede63faa2ce218a58fcbe4465c7'
#       auth_token = '6ae316781c8d00b79f4cc1f792c9ae10'
#       client = Client(account_sid, auth_token)
#
#       message = client.messages \
#                          .create(
#                             body="glaucoma",
#                             from_='+12054028214',
#                             to='+919108545745'
#                           )
#
#       print(message.sid)


plt.style.use('fivethirtyeight')


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
     # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
     # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

plot_model_history(model_info)

