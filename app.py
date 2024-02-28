import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
app = Flask(__name__)

ensemble_model = pickle.load(open('./model/model.pkl', 'rb'))

# ann_model = load_model('./model/model2A.h5')

@app.route('/')
def hello_world():
    return render_template('index.html')


def has_url(acc_desc):
    # Define a regex pattern for detecting URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Use the findall method to search for URLs in the description
    urls = url_pattern.findall(acc_desc)

    # If any URLs are found, return True, otherwise return False
    return bool(urls)


def numerical_ratio(username):
    if len(username) == 0:
        return 0.0  # Avoid division by zero if the username is empty
    numerical_chars = sum(char.isdigit() for char in username)
    return numerical_chars / len(username)


def transform(raw_data):
    account_name = raw_data[0]
    account_id = raw_data[1]
    follower_cnt = raw_data[2]
    following_cnt = raw_data[3]
    post_cnt = raw_data[4]
    is_pvt = raw_data[5]
    has_pfp = raw_data[6]
    acc_desc = raw_data[7]

    # checking for pfp:
    flag1 = 0
    if has_pfp == 'true':
        flag = 1
    else:
        flag = 0

    # checking for pvt:
    flag2 = 0
    if is_pvt == 'true':
        flag2 = 1
    else:
        flag2 = 0

    # final = {
    #     'profile pic': [flag1],
    #     'nums/length username': [float(numerical_ratio(account_id))],
    #     'fullname words': [len(account_name.split())],
    #     'nums/length fullname': [float(numerical_ratio(account_name))],
    #     'name==username': [int(account_name == account_id)],
    #     'description length': [len(acc_desc)],
    #     'external URL': [int(has_url(acc_desc))],
    #     'private': [flag2],
    #     '#posts': [int(post_cnt)],
    #     '#followers': [int(follower_cnt)],
    #     '#follows': [int(following_cnt)]
    # }

    final = [
        flag1,
        float(numerical_ratio(account_id)),
        len(account_name.split()),
        float(numerical_ratio(account_name)),
        int(account_name == account_id),
        len(acc_desc),
        int(has_url(acc_desc)),
        flag2,
        int(post_cnt),
        int(follower_cnt),
        int(following_cnt)
    ]

    return final


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    raw_data = [x for x in request.form.values()]
    print("Raw Data:", raw_data)

    transform_data = transform(raw_data)
    print("Transformed Data:", transform_data)

    # data = pd.DataFrame(transform_data)
    # print("DataFrame:", data)
    #
    # feature_set = data[['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username',
    #                     'description length', 'external URL', 'private', '#posts', '#followers', '#follows']]
    # print("Feature Set:", feature_set)

    # x = np.asarray(feature_set)
    # print("Input for Prediction:", x)

    # model = request.form.get('model')
    #
    # if model == "1":
    #     x = np.array([transform_data])
    #
    #     prediction = svm_model.predict(x)
    #     print(prediction)
    #
    #     prob = svm_model.predict_proba(x)
    #     probability_fake = round(prob[:, 1][0], 4)
    #     probability_real = 1 - probability_fake
    #
    #     probability_real = round(probability_real, 4) * 100
    #     probability_fake = round(probability_fake, 4) * 100
    #
    #     if prediction[0] == 1:
    #         return render_template('index.html', pred=f'The account is likely fake.\nThe probability is {probability_fake:.2f}%')
    #     else:
    #         return render_template('index.html', pred=f'The account is likely real.\nThe probability is {probability_real:.2f}%')
    #
    # else:
    #     x = np.array([transform_data])
    #
    #     # Use predict method to get raw predictions (probabilities)
    #     raw_predictions = ann_model.predict(x)
    #
    #     # Extract probability for the positive class (assuming binary classification)
    #     probability_fake = round(raw_predictions[0, 1], 4)
    #     probability_real = 1 - probability_fake
    #
    #     probability_real = round(probability_real, 4) * 100
    #     probability_fake = round(probability_fake, 4) * 100
    #
    #     if probability_fake > 50:
    #         return render_template('index.html',
    #                                pred=f'The account is likely fake.\nThe probability is {probability_fake:.2f}%')
    #     else:
    #         return render_template('index.html',
    #                                pred=f'The account is likely real.\nThe probability is {probability_real:.2f}%')

    x = np.array([transform_data])

    prediction = ensemble_model.predict(x)
    print(prediction)

    prob = ensemble_model.predict_proba(x)
    probability_fake = round(prob[:, 1][0], 4)
    probability_real = 1 - probability_fake

    probability_real = round(probability_real, 4) * 100
    probability_fake = round(probability_fake, 4) * 100

    if prediction[0] == 1:
        return render_template('index.html',
                               pred=f'The account is fake.\nThe probability is {probability_fake:.2f}%')
    else:
        return render_template('index.html',
                               pred=f'The account is real.\nThe probability is {probability_real:.2f}%')


if __name__ == '__main__':
    app.run(debug=True)