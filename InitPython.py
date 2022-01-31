import os
# import Tomato
import base64
import datetime
import pyodbc as db
import pathlib
from io import BytesIO

conn = db.connect('Driver={SQL Server};'
                      'Server=DESKTOP-J050F58;'
                      'Database=TOMATO;'
                      'Trusted_Connection=yes;')
cursor = conn.cursor()

imgPath = "E:\\ANN\Source\\PlantVillage-Dataset\\raw\\color\\Tomato___Tomato_Yellow_Leaf_Curl_Virus\\"

for files in os.listdir(imgPath) :
    with open(imgPath+files, "rb") as img2str:
        imgType = pathlib.Path(imgPath+files).suffix
        imgType = imgType[1:len(imgType)]
        strImg = base64.b64encode(img2str.read())
    cursor.execute("INSERT INTO PREPROCESS_D (PREPROCESS_H_ID, IMG_FILE, IMG_TYPE, IS_PROCESSED) VALUES ('10', ?, ?,'0')", strImg, imgType)
    conn.commit()
print("Berhasil generate data.")
        
        

# img = Image.open('E:\ANN\Source\PlantVillage-Dataset\raw\color\Tomato___healthy\0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.JPG')
# img.save('E:\ANN\processed\Tomato___healthy\0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.png', 'PNG')
# src = cv.imread('/content/img/sample3.png')

# dst = cv.fastNlMeansDenoisingColored(src, None, 8, 8, 7, 31)


# cv2_imshow(dst)

# gs = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
# cv2_imshow(gs)