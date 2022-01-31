import numpy as np
import cv2 as cv
import pyodbc as db
import base64 as base
import io
import pandas as pd
import math
import os
from PIL import Image
from base64 import decodestring
from pathlib import Path
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# Old Method
# def imgPre1(src, strength=5):
#     # Lakukan denoising terhadap gambar asli
#     dst = cv.fastNlMeansDenoisingColored(src, None, 8, 8, 7, 31)
#     # Ubah gambar denoised menjadi grayscale
#     gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
#     # Lakukan blur terhadap gambar grayscale
#     blur = cv.medianBlur(gray, strength)
#     return blur

# def imgPre2(src, img):
#     # Lakukan Adaptive Thresholding dengan Thresh Gaussian
#     thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 3)
#     # Gambar contours
#     ct, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     srcPoluted = src.copy()
#     cv.drawContours(srcPoluted, ct, -1, (0,0,0), 2, cv.LINE_AA)
#     return srcPoluted

# def auto_canny(src, img, sigma=0.33):
#   v = np.median(img)
#   if (v > 191) :
#     lower = int(max(0, (1.0 - 2.0*sigma) * (255 - v)))
#     upper = int(max(85, (1.0 + 2.0*sigma) * (255 - v)))
#   elif (v > 127) :
#     lower = int(max(0, (1.0 - sigma) * (255 - v)))
#     upper = int(min(255, (1.0 + sigma) * (255 - v)))
#   elif (v < 63)  :
#     lower = int(max(0, (1.0 - 2.0*sigma) * v))
#     upper = int(max(85, (1.0 + 2.0*sigma) * v))
#   else :
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))

#   edges = cv.Canny(img, lower, upper)
#   cv.imshow(edges)

#   white_bg = np.zeros([256,256,3], dtype=np.uint8)
#   white_bg.fill(200)

#   final = cv.bitwise_and(white_bg, src, mask=edges)
#   return final
# End Old Method

def removeConnRegion(src, size):
    output = src.copy()
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(src)
    for i in range(1, nlabels-1):
      region_size = stats[i, 4]
      if region_size < size:
        x0 = stats[i, 0]
        y0 = stats[i, 1]
        x1 = x0 + stats[i, 2]
        y1 = y0 + stats[i, 3]
        for row in range(y0, y1):
          for col in range(x0, x1):
            if labels[row, col] == i:
              output[row, col] = 0

    return output

def preprocessImg(src):
    # Ubah jadi array
    img = np.array(src, dtype='float64')

    blue = np.zeros((img.shape[0], img.shape[1]) , dtype=img.dtype)
    green = np.zeros((img.shape[0], img.shape[1]) , dtype=img.dtype)
    red = np.zeros((img.shape[0], img.shape[1]) , dtype=img.dtype)

    blue[:,:] = img[:,:,0]
    green[:,:] = img[:,:,1]
    red[:,:] = img[:,:,2]

    # Generate Grayscale
    gs = 2 * green - red - blue
    w = gs.min()
    e = gs.max()
    gs = gs - w
    gs = gs / e*255
    gs = np.array(gs, dtype='uint8')

    # Otsu filtering
    ret, th = cv.threshold(gs, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print("Otsu Filtering Threshold bernilai ", ret)
    hole = th.copy()
    cv.floodFill(hole, None, (0,0), 255)
    hole = cv.bitwise_not(hole)
    filledEdge = cv.bitwise_or(th, hole)

    # Image Corrossion
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    eroded = cv.erode(filledEdge, kernel)

    # Hapus Connected Region
    imgOut = removeConnRegion(eroded, 180)

    img[:, :, 2] = imgOut * red
    img[:, :, 1] = imgOut * green
    img[:, :, 0] = imgOut * blue

    return img

# Inisialisasi proses awal untuk training
def init_process():
    conn = db.connect('Driver={SQL Server};'
                      'Server=DESKTOP-J050F58;'
                      'Database=TOMATO;'
                      'Trusted_Connection=yes;')

    cursor = conn.cursor()
    
    # Periksa apakah tabel sudah terbentuk
    cursor.execute('SELECT COUNT(TABLE_NAME) FROM INFORMATION_SCHEMA.TABLES')
    countTables = cursor.fetchone()[0]
    
    countData = 0

    if (countTables == 0) :
        cursor.execute("exec [dbo].[spInitApp]")
    else :
        # Periksa jumlah gambar yang ada dan belum di proses
        cursor.execute('SELECT COUNT(PREPROCESS_D_ID) FROM PREPROCESS_D WHERE IS_PROCESSED=0')
        countData = cursor.fetchone()[0]

    if countData != 0 :
        cursor.execute('SELECT IMG_FILE, PREPROCESS_H_ID, PREPROCESS_D_ID, IMG_TYPE FROM PREPROCESS_D WHERE IS_PROCESSED=0')
        for row in cursor.fetchall() :
            img_data = row[0]
            preprocessHId = row[1]
            preprocessDId = row[2]
            imgType = row[3]
            cursor.execute("SELECT IMG_CATEGORY FROM PREPROCESS_H WHERE PREPROCESS_H_ID = ?", row[1])
            imgCategory = cursor.fetchone()[0]
            imgName = imgCategory + "_decodeImg." + imgType

            image = Image.open(io.BytesIO(base.b64decode(img_data)))

            image.save("E:\\ANN\\processed\\" + imgName, "PNG")

            imgName = imgCategory + "_processedImg.PNG"
            imgRes = preprocessImg(image)
            imgRes = Image.fromarray(np.uint8(imgRes*255))
            imgRes.save("E:\\ANN\\processed\\" + imgName, "PNG")

            with open("E:\\ANN\\processed\\" + imgName, "rb") as img2str : 
                strImg = base.b64encode(img2str.read())

            # Hapus data yang masih statusnya IS_PROCESSED = 0
            cursor.execute("DELETE FROM PREPROCESS_D WHERE PREPROCESS_D_ID = ? AND IS_PROCESSED = 0", preprocessDId)
            conn.commit()

            # Insert data hasil preprocess
            cursor.execute("INSERT INTO PREPROCESS_D (PREPROCESS_H_ID, IMG_FILE, IMG_TYPE, IS_PROCESSED) VALUES (?, ?, 'PNG', '1')", preprocessHId, strImg)
            conn.commit()
        return countData
    else :
        return 0

def getDataframe(mode):
  conn = db.connect('Driver={SQL Server};'
                      'Server=DESKTOP-J050F58;'
                      'Database=TOMATO;'
                      'Trusted_Connection=yes;')

  cursor = conn.cursor()

  imgPath = str(Path.cwd()) + "\\temp\\"

  # Clear folder dahulu
  # for files in os.listdir(imgPath):
  #   os.remove(os.path.join(imgPath, files))

  # Ambil list data
  gsValue = 0.0

  if mode == 'Training' :
    imgPath = str(Path.cwd()) + "\\temp\\training\\"
    # Clear folder dahulu
    # for files in os.listdir(imgPath):
    #   os.remove(os.path.join(imgPath, files))
    # cursor.execute("SELECT CONVERT(DECIMAL(17,2), GS_VALUE) FROM GENERAL_SETTING WHERE GS_CODE = 'TRAIN_VALMOD'")
    # gsValue = cursor.fetchone()[0]
  else :
    imgPath = str(Path.cwd()) + "\\temp\\testing\\"
    # Clear folder dahulu
    # for files in os.listdir(imgPath):
    #   os.remove(os.path.join(imgPath, files))
    # cursor.execute("SELECT CONVERT(DECIMAL(17,2), GS_VALUE) FROM GENERAL_SETTING WHERE GS_CODE = 'TEST_VALMOD'")
    # gsValue = cursor.fetchone()[0]
  
  countData = 0
  cursor.execute('SELECT COUNT(PREPROCESS_D.PREPROCESS_D_ID), PREPROCESS_D.PREPROCESS_H_ID, PREPROCESS_H.IMG_CATEGORY FROM PREPROCESS_D JOIN PREPROCESS_H ON PREPROCESS_D.PREPROCESS_H_ID = PREPROCESS_H.PREPROCESS_H_ID WHERE PREPROCESS_D.IS_PROCESSED=1 GROUP BY PREPROCESS_D.PREPROCESS_H_ID, PREPROCESS_H.IMG_CATEGORY')
  listPreprocessDId = []
  listImgCategory = []
  for row in cursor.fetchall() :
    countData = row[0]
    preprocessHId = row[1]
    imgCategory = row[2]
    halfCount = math.ceil(countData * gsValue)

    if mode == 'Training' :
      cursor.execute("SELECT TOP (?) IMG_FILE, PREPROCESS_D_ID, IMG_TYPE FROM PREPROCESS_D WHERE IS_PROCESSED = 1 AND PREPROCESS_H_ID = ? ORDER BY PREPROCESS_D_ID ASC", halfCount, preprocessHId)
      # for rowInside in cursor.fetchall():
      #   imgFile = rowInside[0]
      #   preprocessDId = rowInside[1]
      #   imgType = rowInside[2]

      #   image = Image.open(io.BytesIO(base.b64decode(imgFile)))
      #   imgName = imgPath + str(imgCategory) + "." +str(preprocessDId) + ".PNG"

      #   image.save(imgName, "PNG")

      #   listImgCategory.append(str(imgCategory))
    else :
      cursor.execute("SELECT TOP (?) IMG_FILE, PREPROCESS_D_ID, IMG_TYPE FROM PREPROCESS_D WHERE IS_PROCESSED = 1 AND PREPROCESS_H_ID = ? ORDER BY PREPROCESS_D_ID DESC", (countData - halfCount), preprocessHId)
      # for rowInside in cursor.fetchall():
      #   imgFile = rowInside[0]
      #   preprocessDId = rowInside[1]
      #   imgType = rowInside[2]

      #   image = Image.open(io.BytesIO(base.b64decode(imgFile)))
      #   imgName = imgPath + str(preprocessDId) + ".PNG"

      #   image.save(imgName, "PNG")

  fileDir = os.listdir(imgPath)

  if mode == 'Training':
    categories = []
    for fileName in fileDir:
      category = fileName.split('.')[0]
      if category == 'HEALTHY':
        categories.append(0)
      elif category == 'SICK_BS':
        categories.append(1)
      elif category == 'SICK_EB':
        categories.append(2)
      elif category == 'SICK_LB':
        categories.append(3)
      elif category == 'SICK_LM':
        categories.append(4)
      elif category == 'SICK_SL':
        categories.append(5)
      elif category == 'SICK_SM':
        categories.append(6)
      elif category == 'SICK_TS':
        categories.append(7)
      elif category == 'SICK_TM':
        categories.append(8)
      elif category == 'SICK_TY':
        categories.append(9)

    df = pd.DataFrame({
      'filename': fileDir,
      'category': categories
    })

    print('Printing Training Dataframe')

    df.to_csv(str(Path.cwd()) + "\\DataFrameTraining.csv", index=False, mode='w')

    # V1
    # df_preprocessId = pd.DataFrame(fileDir)
    # df_imgCategory = pd.DataFrame(listImgCategory)
    # df = pd.concat([df_preprocessId, df_imgCategory], axis = 1, join = 'inner')
    # df.columns = ['filename', 'class']
    # print('Training Dataframe : ', df)
    # df.to_csv("E:/ANN/python/DataFrameTraining.csv", index=False, mode='w')
    return df
  else :
    df = pd.DataFrame({'filename' : fileDir})
    print('Printing Testing Dataframe')

    df.to_csv(str(Path.cwd()) + "\\DataFrameTesting.csv", index=False, mode='w')
    return df

def generateModel():
  model = Sequential()

  # # LeNet-5 Model
  # # Layer 1
  # model.add(Conv2D(20, 5, padding='same',activation='tanh', input_shape=(128,128,3)))
  # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

  # #Layer 2
  # model.add(Conv2D(50, 5, padding='same',activation='tanh'))
  # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

  # #Layer 3
  # model.add(Flatten())
  # model.add(Dense(500, activation='tanh'))
  # model.add(Dense(10, activation='softmax'))
 
  # LeNet-5 Model V2
  # Layer 1
  model.add(Conv2D(20, 5, padding='same',activation='relu', input_shape=(128,128,3)))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

  #Layer 2
  model.add(Conv2D(50, 5, padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

  #Layer 3
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dense(10, activation='softmax'))

  # LeNet-5 Model V3
  # Layer 1
  # model.add(Conv2D(filters=6, kernel_size=(3,3), activation='relu', input_shape=(128,128,1)))
  # model.add(AveragePooling2D())

  #Layer 2
  # model.add(Conv2D(filters=6, kernel_size=(3,3), activation='relu'))
  # model.add(AveragePooling2D())

  #Layer 3
  # model.add(Flatten())
  # model.add(Dense(units=128, activation='relu'))
  # model.add(Dense(units=10, activation='softmax'))


  # Personal Model
  # # Layer 1
  # model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), activation = 'tanh', input_shape=(128,128,3), padding='same'))
  # model.add(BatchNormalization())
  # model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
  # model.add(Dropout(0.25))

  # # Layer 2
  # model.add(Conv2D(64, kernel_size=(5,5), activation = 'tanh', strides=(1,1), padding='same'))
  # model.add(BatchNormalization())
  # model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
  # model.add(Dropout(0.25))

  # # Layer 3
  # model.add(Conv2D(128, kernel_size = (3,3), activation = 'tanh'))
  # model.add(BatchNormalization())
  # model.add(MaxPooling2D(pool_size = (2,2)))
  # model.add(Dropout(0.25))

  # #Layer 4
  # model.add(Flatten())
  # model.add(Dense(512, activation = 'tanh'))
  # model.add(BatchNormalization())
  # model.add(Dropout(0.5))

  # # Layer 5
  # model.add(Dense(10, activation = 'softmax'))
  
  # Compile v1
  model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
  # Compile v2
  # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.summary()

  return model

def initTraining():
  # Generate model
  model = generateModel()
  # Atur training agar tidak terjadi overfitting
  # earlystop = EarlyStopping(patience=10)
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 5, verbose = 1, factor = 0.5, min_lr = 0.001)
  # callbacks = [earlystop, learning_rate_reduction]

  df = getDataframe('Training')
  df["category"] = df["category"].replace(
    {0: 'Healthy', 1: 'Bacterial Spot', 2: 'Early Blight', 3: 'Late Blight',
    4: 'Leaf Mold', 5: 'Septoria Leaf', 6: 'Spider Mites',
    7: 'Target Spot', 8: 'Mosaic Virus', 9: 'Yellow Leaf Curl Virus'})
  train_df, validate_df = train_test_split(df, test_size = 0.20, random_state = 42)
  train_df = train_df.reset_index(drop = True)
  validate_df = validate_df.reset_index(drop = True)

  # Training v2
  # epochs = 20
  # df_file = df['filename']
  # df_cat = df['class']
  # train_batch = tf.data.Dataset.from_tensor_slices((df_file, df_cat))

  # history = model.fit(train_batch, epochs=20)

  # accuracyRes = history.history['accuracy']

  # Training v3
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose = 1, factor = 0.5, min_lr = 0.00001)

  total_train = train_df.shape[0]
  total_validate = validate_df.shape[0]
  batch_size = 15

  train_datagen = ImageDataGenerator(rotation_range = 15, rescale = 1./255, shear_range = 0.1, zoom_range = 0.2, horizontal_flip = True, width_shift_range = 0.1, height_shift_range = 0.1)
  train_generator = train_datagen.flow_from_dataframe(train_df, str(Path.cwd()) + "\\temp\\training\\", x_col = 'filename', y_col = 'category', target_size = (128, 128), class_mode = 'categorical', batch_size = batch_size)

  validator_datagen = ImageDataGenerator(rescale = 1./255)
  validator_generator = validator_datagen.flow_from_dataframe(train_df, str(Path.cwd()) + "\\temp\\training\\", x_col = 'filename', y_col = 'category', target_size = (128, 128), class_mode = 'categorical', batch_size = batch_size)

  epochs = 20
  # history = model.fit_generator(train_generator, epochs = epochs, validation_data = validator_generator, validation_steps = total_validate // batch_size, steps_per_epoch = total_train // batch_size, callbacks = [learning_rate_reduction])

  # model.save_weights("model_weights_relu_v3.h5")
  # model.save("mainmodel_relu_v3.h5")

  # Training v1
  # total_train = train_df.shape[0]
  # total_validate = validate_df.shape[0]
  # batch_size = 32

  # train_datagen = ImageDataGenerator(rotation_range = 15, rescale = 1./255, shear_range = 0.1, zoom_range = 0.2, horizontal_flip = True, width_shift_range = 0.1, height_shift_range = 0.1)
  # train_generator = train_datagen.flow_from_dataframe(train_df, 'E:/ANN/python/temp/training/', target_size = (128, 128), batch_size = 32)

  # validator_datagen = ImageDataGenerator(rescale = 1./255)
  # validator_generator = validator_datagen.flow_from_dataframe(validate_df, 'E:/ANN/python/temp/training/', target_size = (128, 128), batch_size = 32)

  # epochs = 50

  # history = model.fit(train_generator, steps_per_epoch=total_train//batch_size, epochs = epochs, validation_data = validator_generator, validation_steps = total_validate//batch_size, callbacks = [learning_rate_reduction])

  # history = model.fit_generator(train_generator, epochs = epochs, validation_data = validator_generator, validation_steps = total_validate//batch_size, steps_per_epoch = total_train//batch_size, callbacks = [learning_rate_reduction])

  # model.save_weights("model_weights_relu.h5")
  # model.save("mainmodel_relu.h5")

  # accuracyRes = max(history.history['accuracy'])

  # accuracyRes = model.evaluate(train_generator, verbose=0)

  # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,12))
  # ax1.plot(history.history['loss'], color='b', label="Training Loss")
  # ax1.plot(history.history['val_loss'], color='r', label="Validation Loss")
  # ax1.set_xticks(np.arange(1, epochs, 1))
  # ax1.set_yticks(np.arange(0, 1, 0.1))

  # ax2.plot(history.history['accuracy'], color='b', label="Training Accuracy")
  # ax2.plot(history.history['val_accuracy'], color='r', label="Validation Accuracy")
  # ax2.set_xticks(np.arange(1, epochs, 1))

  # legend = plt.legend(loc='best', shadow=True)
  # plt.tight_layout()
  # plt.show(block=False)

  conn = db.connect('Driver={SQL Server};'
                      'Server=DESKTOP-J050F58;'
                      'Database=TOMATO;'
                      'Trusted_Connection=yes;')

  cursor = conn.cursor()

  # cursor.execute("UPDATE GENERAL_SETTING SET GS_VALUE = ? WHERE GS_CODE = ?", str(accuracyRes), "TRAIN_ACC")

  print("Training Complete")

  # model = tf.keras.models.load_model("mainmodel_relu.h5")
  model = tf.keras.models.load_model("mainmodel_relu_v3.h5")
  # model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

  test_df = getDataframe('Test')
  sample_count = df.shape[0]

  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_dataframe(test_df, str(Path.cwd()) + "\\temp\\testing\\", x_col = 'filename', y_col = None, class_mode = None, target_size = (128, 128), batch_size = 15, shuffle = False)


  predict = model.predict_generator(test_generator, steps = np.ceil(sample_count/15))
  # predict = model.predict(test_generator, steps = np.ceil(sample_count / 15))

  test_df['category'] = np.argmax(predict, axis = -1)

  label_map = dict((v,k) for k,v in train_generator.class_indices.items())
  test_df['category'] = test_df['category'].replace(label_map)

  sample_test = test_df.head(3)
  sample_test.head()
  plt.figure(figsize=(12,24))
  for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(str(Path.cwd()) + "\\temp\\testing\\" + filename, target_size = (128,128))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
  plt.tight_layout()
  plt.show()

  print("Testing Complete")
  return "Model trained and tested"

def initTesting():
  # model = tf.keras.models.load_model("mainmodel_relu.h5")
  model = tf.keras.models.load_model("mainmodel_relu_v3")
  # model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

  df = getDataframe('Test')
  sample_count = df.shape[0]

  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_dataframe(df, str(Path.cwd()) + "\\temp\\testing\\", x_col = 'filename', y_col = None, class_mode = None, target_size = (128, 128), batch_size = 15, shuffle = False)


  predict = model.predict_generator(test_generator, steps = np.ceil(sample_count/15))
  # predict = model.predict(test_generator, steps = np.ceil(sample_count / 15))

  df['category'] = np.argmax(predict, axis = -1)

  label_map = dict((v,k) for v,k in train_generator.class_indices.items())
  df['category'] = df['category'].replace(label_map)

  sample_test = df.head(3)
  sample_test.head()
  plt.figure(figsize=(12,24))
  for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(str(Path.cwd()) + "\\temp\\testing\\" + filename, target_size = (128,128))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
  plt.tight_layout()
  plt.show()

  accuracyRes = model.evaluate(test_generator, verbose=0)

  conn = db.connect('Driver={SQL Server};'
                      'Server=DESKTOP-J050F58;'
                      'Database=TOMATO;'
                      'Trusted_Connection=yes;')

  cursor = conn.cursor()

  cursor.execute("UPDATE GENERAL_SETTING SET GS_VALUE = ? WHERE GS_CODE = ?", str(accuracyRes), "TEST_ACC")

  return "Testing Complete"

def initUji():
  try:
    # model = tf.keras.models.load_model("mainmodel_relu.h5")
    model = tf.keras.models.load_model("mainmodel_relu_v3.h5")

    conn = db.connect('Driver={SQL Server};'
                        'Server=DESKTOP-J050F58;'
                        'Database=TOMATO;'
                        'Trusted_Connection=yes;')

    cursor = conn.cursor()

    img = image.load_img(str(Path.cwd()) + "\\temp\\rawUser.png", target_size = (128, 128))

    strImg = b'\xff'

    with open(str(Path.cwd()) + "\\temp\\rawUser.png", "rb") as f :
      strImg = base.b64encode(f.read())

    img = preprocessImg(img)

    with open(str(Path.cwd()) + "\\temp\\processedUser.png", "wb") as f :
      f.write(img)
      f.close()

    # img = image.load_img('E:/ANN/python/temp/19013.png', target_size = (128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    images = np.vstack([x])
    classes = np.argmax(model.predict(images), axis=-1)
    # classes = model.predict_classes(images, batch_size = 15)
    preprocessHId = classes[0]
    imgCategory = ''
    if preprocessHId == 0:
      imgCategory = 'Tidak diketahui'
    else:
      cursor.execute("SELECT CATEGORY_DETAIL FROM PREPROCESS_H WHERE PREPROCESS_H_ID = ?", int(preprocessHId))
      imgCategory = cursor.fetchone()[0]

    cursor.execute("SELECT GS_VALUE FROM GENERAL_SETTING WHERE GS_CODE = ?", "TEST_ACC")
    acc = cursor.fetchone()[0]

    cursor.execute("INSERT INTO UJI_RESULT (IMG_FILE, IMG_TYPE, PREDICT_CATEGORY) VALUES (?, ?, ?)", strImg, "PNG", int(preprocessHId))

    print('Kategori gambar : ', imgCategory)
    print('Akurasi Model: ', acc)
  except:
    imgCategory = 'ERR'
    acc='100%'

  return imgCategory, acc