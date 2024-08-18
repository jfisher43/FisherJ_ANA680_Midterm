from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize the flask app
app = Flask(__name__)

# Load the model
filename = 'midterm_model.pkl'
model = pickle.load(open(filename, 'rb'))

# column_names = ['gender', 'parental level of education', 'lunch', 'test preparation course', 'math score', 'reading score', 'writing score']
# input_data = pd.DataFrame([[1, 1, 1, 1, 72, 72, 74]], columns=column_names)
# prediction = model.predict(input_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['gender'])
        parent_education = int(request.form['parent_education'])
        lunch = int(request.form['lunch'])
        test_prep_course = int(request.form['test_prep_course'])
        math_score = int(request.form['math_score'])
        reading_score = int(request.form['reading_score'])
        writing_score = int(request.form['writing_score'])

        input_data = np.array([[gender, parent_education, lunch, test_prep_course, math_score, reading_score, writing_score]])
        print("Input data for prediction:", input_data)

        pred = model.predict(input_data)
        print("Prediction result:", pred)

        # Mapping the prediction to the appropriate label
        if pred[0] == 0:
            result = "Group A"
        elif pred[0] == 1:
            result = "Group B"
        elif pred[0] == 2:
            result = "Group C"
        elif pred[0] == 3:
            result = "Group D"
        elif pred[0] == 4:
            result = "Group E"
        else:
            result = "Unknown" 

        return render_template('index.html', predict=result)
    except Exception as e:
        print("Error occurred:", e)
        return render_template('index.html', predict="An error occurred. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
