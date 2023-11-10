import pandas as pd
from flask import Flask, render_template, request
import re
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('./model/model.pkl', 'rb'))


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

    final = {
        'profile pic': [flag1],
        'nums/length username': [float(len(account_id) / 100)],
        'fullname words': [len(account_name.split())],
        'nums/length fullname': [float(len(account_name) / 100)],
        'name==username': [int(account_name == account_id)],
        'description length': [len(acc_desc)],
        'external URL': [int(has_url(acc_desc))],
        'private': [flag2],
        '#posts': [int(post_cnt)],
        '#followers': [int(follower_cnt)],
        '#follows': [int(following_cnt)]
    }

    return final


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    raw_data = [x for x in request.form.values()]
    print("Raw Data:", raw_data)

    transform_data = transform(raw_data)
    print("Transformed Data:", transform_data)

    data = pd.DataFrame(transform_data)
    print("DataFrame:", data)

    feature_set = data[['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username',
                        'description length', 'external URL', 'private', '#posts', '#followers', '#follows']]
    print("Feature Set:", feature_set)

    x = np.asarray(feature_set)
    print("Input for Prediction:", x)

    prediction = model.predict(x)
    print("Raw Prediction:", prediction)

    prob = model.predict_proba(x)
    # print("Percentage:", percentage)

    probability_fake = prob[:, 1][0]
    probability_real = 1 - probability_fake

    probability_real = round(probability_real, 4) * 100
    probability_fake = round(probability_fake, 4) * 100

    if prediction[0] == 1:
        return render_template('index.html', pred=f'The account is likely fake.\nThe probability is {probability_fake:.2f}%')
    else:
        return render_template('index.html', pred=f'The account is likely real.\nThe probability is {probability_real:.2f}%')



if __name__ == '__main__':
    app.run(debug=True)