
from flask import Flask, render_template, url_for, request, redirect, session
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from wtforms import FloatField,StringField,SubmitField,SelectField
from wtforms.validators import DataRequired, InputRequired,NumberRange


import pandas as pd
import numpy as np
import sklearn
import joblib


app = Flask(__name__)
Bootstrap(app)
app.config["SECRET_KEY"] = "hard to guess string"


@app.route('/')
def index():
    return render_template("base.html")

@app.route('/description')
def description():
	return render_template("summary.html")



class Features(FlaskForm):
	# Flask Form to enter the different parameters
	height = FloatField( 'Enter height value:',validators=[DataRequired(message = 'data required'),NumberRange(min=1,max=804)])
	area = FloatField( 'Enter area value:',validators=[DataRequired(message = 'data required'),NumberRange(min=7,max=143993)])
	eccen = FloatField( 'Enter eccen value:',validators=[DataRequired(message = 'data required'),NumberRange(min=0.007,max=537)])
	p_black = FloatField( 'Enter p_black value',validators=[DataRequired(message = 'data required'),NumberRange(min=0.052,max=1)])
	p_and = FloatField( 'Enter p_and:',validators=[DataRequired(message = 'data required'),NumberRange(min=0.062,max=1)])
	blackpix = FloatField( 'Enter blackpix value',validators=[DataRequired(message = 'data required'),NumberRange(min=7,max=33017)])
	blackand = FloatField( 'Enter blackand value',validators=[DataRequired(message = 'data required'),NumberRange(min=7,max=46133)])
	model_choice = SelectField( 'Enter your model choice:',choices = [('Decision_Tree','decision tree'),('Random_Forest','random forest'),('Bagging','bagging')])
	submit = SubmitField()

@app.route('/predict', methods = ["GET","POST"])
def Prediction():

	form = Features()
	if request.method == "POST" and form.validate_on_submit():

	  #Get the parameters from the form
	  parameters_data = [form.height.data, form.area.data, form.eccen.data, form.p_black.data, form.p_and.data, form.blackpix.data, form.blackand.data]
	  parameters = np.array(parameters_data).reshape(1,-1)

	  # Loading the Model and get the prediction
	  if form.model_choice.data == 'Decision_Tree':
		  decision_tree_model = joblib.load('models_and_data/decision_tree_model.pkl')
		  result_prediction = decision_tree_model.predict(parameters)
	         
	  elif form.model_choice.data == 'Random_Forest':
		  random_forest_model = joblib.load('models_and_data/random_forest_model.pkl')
		  result_prediction = random_forest_model.predict(parameters)
		    
	  
	  elif form.model_choice.data == 'Bagging':
		  bagging_model = joblib.load('models_and_data/bagging_model.pkl')
		  result_prediction = bagging_model.predict(parameters)
	
	  result = result_prediction

	  #Store it in a session to get the value after
	  session["results"] = str(result[0])
	
	return render_template('prediction.html', form = form)


@app.route('/results')
def show_results():
	#show the result prediction and to what it corresponds
	pred = session["results"]
	Page_block = ["Text", "Horizontal Line", "Graphic", "Vertical Line", "Picture"]
	pageblock = Page_block[int(pred)-1]
	
	return render_template ('results.html', pred = pred, pageblock=pageblock)


if __name__ == '__main__':
	app.run(host = 'localhost',port=4000, debug=True)









