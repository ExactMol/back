from flask import Flask, request, jsonify, send_from_directory
import os
import joblib
import sklearn
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
app = Flask(__name__)


def get_predictor():
    return joblib.load("predict1.sav")


# def predict(model: str, query: str) -> str:
#     # TODO model code
#     return 'jalal rabota sdelana'


@app.route('/api/v1/predict', methods=['POST'])
def predictEndpoint():
    data = request.json
    model = data.get('model') if data else None
    query_data = data.get('query') if data else None
    mol = Chem.MolFromSmiles(query_data)
    print(sklearn.__version__)
    descriptors = list(np.array(Descriptors._descList)[:, 0])
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
    result = np.array(calculator.CalcDescriptors(mol))
    cleared_result = []
    for i in result:
        if i != 0:
            cleared_result.append(i)
    predictor = get_predictor()
    predicted_result = predictor.predict([result])

    response = {
        "key": predicted_result.tolist()
    }
    return jsonify(response)


@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run(debug=True)
