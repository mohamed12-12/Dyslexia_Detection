from flask import Flask, request, jsonify
import pandas as pd
import pickle

MODEL_FEATURES = [
    'age', 'clicks1', 'hits1', 'misses1', 'score1', 'accuracy1', 'missrate1',
    'clicks2', 'hits2', 'misses2', 'score2', 'accuracy2', 'missrate2',
    'clicks3', 'hits3', 'misses3', 'score3', 'accuracy3', 'missrate3',
    'clicks4', 'hits4', 'misses4', 'score4', 'accuracy4', 'missrate4',
    'clicks5', 'hits5', 'misses5', 'score5', 'accuracy5', 'missrate5',
    'clicks6', 'hits6', 'misses6', 'score6', 'accuracy6', 'missrate6',
    'clicks7', 'hits7', 'misses7', 'score7', 'accuracy7', 'missrate7',
    'clicks8', 'hits8', 'misses8', 'score8', 'accuracy8', 'missrate8',
    'clicks9', 'hits9', 'misses9', 'score9', 'accuracy9', 'missrate9',
    'clicks10', 'hits10', 'misses10', 'score10', 'accuracy10', 'missrate10',
    'clicks11', 'hits11', 'misses11', 'score11', 'accuracy11', 'missrate11',
    'clicks12', 'hits12', 'misses12', 'score12', 'accuracy12', 'missrate12',
    'clicks13', 'hits13', 'misses13', 'score13', 'accuracy13', 'missrate13',
    'clicks14', 'hits14', 'misses14', 'score14', 'accuracy14', 'missrate14',
    'clicks15', 'hits15', 'misses15', 'score15', 'accuracy15', 'missrate15',
    'clicks16', 'hits16', 'misses16', 'score16', 'accuracy16', 'missrate16',
    'clicks17', 'hits17', 'misses17', 'score17', 'accuracy17', 'missrate17',
    'clicks18', 'hits18', 'misses18', 'score18', 'accuracy18', 'missrate18',
    'clicks19', 'hits19', 'misses19', 'score19', 'accuracy19', 'missrate19',
    'clicks20', 'hits20', 'misses20', 'score20', 'accuracy20', 'missrate20',
    'clicks21', 'hits21', 'misses21', 'score21', 'accuracy21', 'missrate21',
    'clicks22', 'hits22', 'misses22', 'score22', 'accuracy22', 'missrate22',
    'clicks23', 'hits23', 'misses23', 'score23', 'accuracy23', 'missrate23',
    'clicks24', 'hits24', 'misses24', 'score24', 'accuracy24', 'missrate24',
    'clicks25', 'hits25', 'misses25', 'score25', 'accuracy25', 'missrate25',
    'clicks26', 'hits26', 'misses26', 'score26', 'accuracy26', 'missrate26',
    'clicks27', 'hits27', 'misses27', 'score27', 'accuracy27', 'missrate27',
    'clicks28', 'hits28', 'misses28', 'score28', 'accuracy28', 'missrate28',
    'clicks29', 'hits29', 'misses29', 'score29', 'accuracy29', 'missrate29',
    'clicks30', 'hits30', 'misses30', 'score30', 'accuracy30', 'missrate30',
    'clicks31', 'hits31', 'misses31', 'score31', 'accuracy31', 'missrate31',
    'clicks32', 'hits32', 'misses32', 'score32', 'accuracy32', 'missrate32',
    'gender_Male', 'nativelang_Yes', 'otherlang_Yes'
]

# Load the trained model once at startup
def load_model():
    model_path = 'Saved_Model_Status/beforeSelectedColumns_ExtraTreesClassifier'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert to DataFrame and ensure correct feature order
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=MODEL_FEATURES, fill_value=0)

        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)