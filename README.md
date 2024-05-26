# FAKE USER PROFILE IDENTIFICATION SYSTEM USING ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING

## Project Description:

The Fake User Profile Identification System is a machine learning-based application designed to identify whether an Instagram profile is genuine or fake. This project utilizes advanced algorithms such as Support Vector Machines (SVM) and Artificial Neural Networks (ANN) to analyze user profile data and predict the authenticity of the account.

## Features
User Profile Information Analysis: Input various profile attributes to determine if the account is fake or real.
- Machine Learning Models: Utilizes SVM and ANN for accurate prediction.
- Web Interface: Simple and user-friendly web interface built using Flask.

## Requirements
- Flask==3.0.3
- joblib==1.3.2
- numpy==1.26.4
- pandas==2.2.2
- scikit_learn==1.3.2
- tensorflow==2.15.0
- tensorflow_intel==2.15.0

## Installation:

### 1. Clone the repository:

    git clone https://github.com/yourusername/fake-user-profile-identification.git
    cd fake-user-profile-identification

### 2. Create a virtual environment:

    python -m venv venv
    source venv/bin/activate    # On Windows, use `venv\Scripts\activate`

### 3. Install the required packages:

    pip install -r requirements.txt

Ensure requirements.txt contains the following:

- Flask==3.0.3
- joblib==1.3.2
- numpy==1.26.4
- pandas==2.2.2
- scikit_learn==1.3.2
- tensorflow==2.15.0
- tensorflow_intel==2.15.0

## Usage:

### Run the Flask application:

    python app.py

### Open your web browser and navigate to:

    http://127.0.0.1:5000/

### Entering user profile information:

After entering all the profile details, click on the "Predict" button to see whether the profile is identified as real or fake based on the input data.

## Training the Model:
If you wish to train the model with new data:

- Prepare the dataset: Ensure it is in a CSV format with appropriate labels.
- Train the models: `python train_model.py`
- Save the trained models in the models/ directory.

## Contributing:

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss any changes.

## Acknowledgements:

The datasets used for training were sourced from Kaggle.
Inspiration and algorithms were drawn from various academic papers and online resources.

## License
This project is licensed under the MIT License.
