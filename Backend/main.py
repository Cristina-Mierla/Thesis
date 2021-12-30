# import main Flask class and request object
import json

from flask import Flask, request, send_file
from service import Service

# create the Flask app
app = Flask(__name__)
service = Service("csv_dataset.csv")
# prediction_model = Model()


@app.route('/patients', methods=['GET'])
def getPatients():
    if request:
        patientList = service.getPatientHeadList()
        result = json.dumps(patientList)

        return result, 200
    else:
        return "Invalid request", 400


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
