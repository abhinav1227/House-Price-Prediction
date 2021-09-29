import pickle
import pandas as pd
from flask import Flask, request, render_template


app = Flask('House_prediction')

#loading our model
with open('model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    df_test = request.form['file']
    df_test = pd.read_csv(df_test)
    prediction = model.predict(df_test)
    output = str(list(prediction))

    return render_template('index.html', prediction_text=f'House Prices for the given list is {output}')


if __name__ == '__main__':
    app.run(debug=True)
