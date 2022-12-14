from flask import Flask, request, render_template, url_for 
import pandas as pd
import tensorflow as tf
import random as rand
import keras

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])

# подаем пользователю на выбор в выпадающее меню строки из dataframe (csv) для выбора, отправки в модель и дальнешйего предсказания
def dropdown():
    print(request.method)
    model = tf.keras.models.load_model(r"NNmodel_mlp")
    df = pd.read_csv("test_ff.csv")
    y_pred = 0
    if request.method=='POST': 
        select = request.form['operator'] #post('operator')
        select_splitted = select.strip('[').strip(']').split()   
        select_splitted_to_pdseries = pd.Series(select_splitted)
        select_splitted_to_pdseries_list=[]
        for i in select_splitted_to_pdseries:
            select_splitted_to_pdseries_list.append(float(i))
        #y_pred = model.predict([[select_splitted_to_pdseries]])
        select_splitted_to_pdseries = pd.Series(select_splitted_to_pdseries_list)
        to_model = select_splitted_to_pdseries.values.reshape(1,-1)
        y_pred = model.predict(to_model)
    return render_template('main.html', df = df, result=y_pred)

def main():
    if request.method == 'GET':
        return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)