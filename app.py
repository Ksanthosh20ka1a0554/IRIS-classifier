from flask import Flask,request,render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

iris = pd.read_csv('Iris dataset.csv')
iris.shape
iris.head()
iris.describe()

columns = [col_name for col_name in iris.columns]
features = columns[:4]
classes = {0:'setosa',1:'versicolor',2:'virginica'}

X = iris[features].values
Y = iris['species'].values

lablel_encoder = LabelEncoder()
Y = lablel_encoder.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')

@app.route("/review",methods=['GET','POST'])
def predict():
    if request.method=="POST":
         sep_length=float(request.form['s_length'])
         sep_width=float(request.form['s_width'])
         pet_length=float(request.form['p_length'])
         pet_width=float(request.form['p_width'])
         pre_list = [sep_length,sep_width,pet_length,pet_width]
         pred_value = knn.predict([pre_list])[0]
         flower = classes[pred_value]
         if flower=='setosa':
            src='iris_setosa.png'
         elif flower=='versicolor':
            src='iris_versicolor.png'
         elif flower=='virginica':
            src='iris_virginica.png'

    return render_template('index.html',final_result_image=src,show_image=True)
    




# from flask import Flask, redirect, url_for
# app = Flask(__name__)

# @app.route('/admin')
# def hello_admin():
#    return 'Hello Admin'

# @app.route('/guest/<guest>')
# def hello_guest(guest):
#    return 'Hello %s as Guest' % guest

# @app.route('/user/<name>')
# def hello_user(name):
#    if name =='admin':
#       return redirect(url_for('hello_admin'))
#    else:
#       return redirect(url_for('hello_guest',guest = name))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)