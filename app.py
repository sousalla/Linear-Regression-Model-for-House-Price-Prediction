from flask import Flask, render_template, request
import joblib as pickle
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle de régression linéaire
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Récupération des données du formulaire
        surface = float(request.form['Surface'])
        chambres = int(request.form['Chambres'])
        annee_construction = int(request.form['AnneeConstruction'])
        distance_centre_ville = float(request.form['DistanceCentreVille'])
        qualite_quartier = int(request.form['QualiteQuartier'])

        # Conversion des données en tableau numpy
        data = np.array([[surface, chambres, annee_construction, distance_centre_ville, qualite_quartier]])

        # Prédiction du prix
        prediction = model.predict(data)
        output = round(prediction[0], 2)

        return render_template('result.html', prediction_text=f'Le prix prédit de la maison est {output} MAD.')

if __name__ == '__main__':
    app.run(debug=True)
