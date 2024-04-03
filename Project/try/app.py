import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load the trained model
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect features from form data
    statuses_count=int(request.form['statuses_count'])
    followers_count=float(request.form['followers_count'])
    friends_count=float(request.form['friends_count'])
    listed_count=int(request.form['listed_count'])
    favourites_count=int(request.form['favourites_count'])
    lang=int(request.form['lang'])
    default_profile=int(request.form['default_profile'])
    profile_use_background_image=int(request.form['profile_use_background_image'])
    

    
    
    feature_array=[[statuses_count,followers_count,friends_count,listed_count,favourites_count,lang,default_profile,
                     profile_use_background_image]]
    #print(features_array)


    
    '''encoded_features = []
    for feature in features:
        if isinstance(feature, str):
            print("string")
            encoded_feature = le.fit_transform([feature])
            encoded_features.append(encoded_feature[0])
        else:
            encoded_features.append(feature)
    features_array = np.array(encoded_features).reshape(1, -1)
    print(features_array)
    '''
    features_array = np.array(feature_array).reshape(1, -1)
    print(feature_array)
    prediction = model.predict(features_array)[0]
    print(prediction)

    # Output the prediction
    if prediction == 1:
        return render_template("fake.html", prediction="Fake Profile...!") 
    else:
        return render_template("real.html", prediction="Real Profile...!")

    
if __name__ == "__main__":
    app.run(debug=True,port=5050)
