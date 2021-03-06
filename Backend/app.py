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


@app.route('/stat4.1', methods=['GET'])
def getStatisticsMultiple1():
    if request:

        image1, image2 = service.getStatistics4()

        return send_file(image1, mimetype='image/png'), 200
    else:

        return "Invalid request", 400


@app.route('/stat4.2', methods=['GET'])
def getStatisticsMultiple2():
    if request:

        image1, image2 = service.getStatistics4()

        return send_file(image2, mimetype='image/png'), 200
    else:

        return "Invalid request", 400


@app.route('/prediction', methods=['POST'])
def getPredict():
    if request:
        json_result = request.get_json(force=True)
        print(json_result)
        age = int(json_result["Age"])
        sex = json_result["Gender"]
        diagnos_int = json_result["Diag"]
        spitalizare = int(json_result["Hosp"])
        ati = int(json_result["Icu"])
        medication = json_result['Med'][0]
        analize = json_result['Anlz'][0]
        comorb = json_result['Comb']
        id = json_result['Id']

        prediction_data = [age, sex, diagnos_int, spitalizare, ati, analize, medication, comorb, id]

        prediction_result = service.makePrediction(prediction_data)

        label = {0: 'Cured', 1: 'Improved', 2: 'Stationary', 3: 'Worsened', 4: 'Deceased'}

        pred = label[prediction_result.tolist().index(max(prediction_result))]
        result = {0: 'Cured', 1: 'Improved', 2: 'Stationary', 3: 'Worsened', 4: 'Deceased'}
        printed_result = "The patient has a high chance to be release as: {}\n\n".format(pred)

        for i in range(0, 5):
            aux = "\t{:20} -> {}%\n".format(label[i], round(prediction_result[i] * 100, 0))
            printed_result = printed_result + aux
            result[i] = prediction_result[i]

        print(printed_result)
        result_json = json.dumps(str(printed_result))

        return result_json, 200

    else:

        return "Invalid request", 400


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
