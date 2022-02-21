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


@app.route('/prediction', methods=['PUT'])
def getPredict():
    if request:
        age = int(request.args.get("Age"))
        sex = int(request.args.get("Gender"))
        diagnos_int = request.args.get("Admission_diagnostic")
        spitalizare = int(request.args.get("Hospitalization"))
        ati = int(request.args.get("ATI"))
        analize = request.args.get('Analyzes')
        comorb = request.args.get('Comorbidities')
        id = int(request.args.get('Id'))

        prediction_data = [age, sex, diagnos_int, spitalizare, ati, analize, comorb, id]

        prediction_result, prediction_percentage = service.makePrediction(prediction_data)

        label = {0: 'Cured', 1: 'Improved', 2: 'Stationary', 3: 'Worsened', 4: 'Deceased'}

        result = "The patient has a high chance to be release as: {}\n\n".format(label[prediction_result[0]])

        for i in range(0, 5):
            aux = "\t{:20} -> {}%\n".format(label[i], round(prediction_percentage[0][i] * 100, 0))
            result = result + aux

        print(result)

        return result, 200

    else:

        return "Invalid request", 400


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
