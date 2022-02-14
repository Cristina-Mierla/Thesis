# import main Flask class and request object
import json

from flask import Flask, request, send_file
from service import Service

# create the Flask app
app = Flask(__name__)
service = Service("csv_processedDataset.csv")
# prediction_model = Model()


@app.route('/patients', methods=['GET'])
def getPatients():
    if request:
        patientList = service.getPatientHeadList()
        result = json.dumps(patientList)

        return result, 200
    else:
        return "Invalid request", 400


@app.route('/patientId', methods=['GET'])
def getPatientById():
    if request:
        patientId = int(request.args.get("patient_id"))
        patientList = service.getPatientById(patientId)
        result = json.dumps(patientList)

        return result, 200
    else:
        return "Invalid request", 400


@app.route('/stat1', methods=['GET'])
def getStatistic1():
    if request:
        image = service.getStatistics1()

        return send_file(image, mimetype='image/png'), 200
    else:

        return "Invalid request", 400


@app.route('/stat2', methods=['GET'])
def getStatistic2():
    if request:
        image = service.getStatistics2()

        return send_file(image, mimetype='image/png'), 200
    else:

        return "Invalid request", 400


@app.route('/stat3', methods=['GET', 'POST'])
def getStatistic3():
    if request:
        age = int(request.args.get("age"))
        gender = int(request.args.get("gender"))
        image = service.getStatistics3(age, gender)

        return send_file(image, mimetype='image/png'), 200
    else:

        return "Invalid request", 400


# @app.route('/prediction', methods=['PUT'])
# def getPredict():
#     if request:
#         form = request.form
#
#         age = json.loads(form["RiskPred"])["Age"]
#         sex = json.loads(form["RiskPred"])["Gender"]
#         diagnos_int = json.loads(form["RiskPred"])["Admission_diagnostic"]
#         spitalizare = json.loads(form["RiskPred"])["Hospitalization"]
#         ati = json.loads(form["RiskPred"])["ATI"]
#         analize = json.loads(form['RiskPred'])['Analyzes']
#         comorb = json.loads(form['RiskPred'])['Comorbidities']
#         id = json.loads(form['RiskPred'])['Id']


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
