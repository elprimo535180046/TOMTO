from flask import Flask, render_template, request, redirect, session, jsonify
from flask_cors import CORS
from pathlib import Path
import json

import Tomato

app = Flask(__name__)
CORS(app)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/jenisPenyakit")
def jenisPenyakit():
    return render_template('jenisPenyakit.html')

@app.route("/uji")
def uji():
    return render_template('uji.html')

@app.route("/DCAWOcwad815")
def DCAWOcwad815():
    # countData = Tomato.init_process()
    # if countData != 0 :
    #     print(countData, " berhasil diproses.")
    # else :
    #     print ("Tidak ada data yang perlu di proses.")

    trainResult = Tomato.initTraining()
    print(trainResult)

    # testResult = Tomato.initTesting()
    # print(testResult)

@app.route("/uji/action", methods=["POST"])
def ujiAction():
    print('File request submitted')

    print(request.files)
    reqFile = request.files['imgTmt']
    print(reqFile)

    for i in request.files.getlist('imgTmt'):
        bytesImage = i.stream.read()

        with open(str(Path.cwd()) + "\\temp\\rawUser.png", "wb") as f:
            f.write(bytesImage)
            f.close()
        
        with open(str(Path.cwd()) + "\\static\\img\\rawUser.png", "wb") as f:
            f.write(bytesImage)
            f.close()

    ujiResult, accResult = Tomato.initUji()
    return ujiResult, accResult

    # filedata = request.files.getlist('imageInput')
    # print(filedata)

       
    # result = "fileImage="+filedata+"&res="
    # resJson = json.dumps(result)
    # return resJson


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
# ================================================
