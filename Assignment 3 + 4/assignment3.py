import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def hardCodeGradientDescent(columns):
	learn_rate=0.0001
	number_iter=100
	initial_weights=np.random.normal(size=(columns,1))
	return learn_rate,number_iter,initial_weights


def hardCodeTraining(rows):
	percent_training=0.8
	number_of_dataPoints=rows
	return math.floor(percent_training*number_of_dataPoints)

def getData():
	fileName="./Folds5x2_pp.xlsx"
	column_headers=["Temperature","Vacuum","Pressure","Humidity","Power"]
	tempFrame = pd.read_excel(fileName,sheet_name=0,names=column_headers)
	return tempFrame

def predict(weights,inp):
	return np.matmul(inp,weights) # weights is a column matrix

def MLE(dataset):
	rows=dataset.shape[0]
	columns=dataset.shape[1]
	y=dataset[:,-1][np.newaxis] # get the last column containing the power consumption data
	y=np.transpose(y) # as the previous gives a 2D row vector
	dataset=dataset[:,:-1]

	one=np.ones((rows,1))
	aug_mat=np.hstack((one,dataset)) # adding the column of ones

	prod=np.matmul(np.transpose(aug_mat),aug_mat)
	intermediate=np.matmul(np.linalg.inv(prod),np.transpose(aug_mat))
	weights=np.matmul(intermediate,y)
	np.transpose(weights)
	return weights

def gradDes(dataset):
	rows=dataset.shape[0]
	columns=dataset.shape[1]
	one=np.ones((rows,1))
	dataset=np.hstack((one,dataset))
	learn_rate,number_iter,weights=hardCodeGradientDescent(columns)
	y=dataset[:,-1][np.newaxis] # get the last column containing the power consumption data
	y=np.transpose(y)
	grad=np.zeros( ((columns),1) )
	dataset=dataset[:,:-1]
	# print(dataset.shape)
	# print(dataset[:10,:])

	for i in range(number_iter):
		y_predicted=predict(weights,dataset)
		for j in range(columns): # we exclude the last as it has the dependent variable
			grad[j]=np.matmul(np.transpose(y_predicted-y),dataset[:,j]) # ycap-y multiplied by x
			weights[j]-=learn_rate*grad[j] # we make use of the older values of w to get the new ones

	return weights

def ridge(dataset,lmbda):
	rows=dataset.shape[0]
	columns=dataset.shape[1]
	one=np.ones((rows,1))
	dataset=np.hstack((one,dataset))
	learn_rate,number_iter,weights=hardCodeGradientDescent(columns)
	y=dataset[:,-1][np.newaxis] # get the last column containing the power consumption data
	y=np.transpose(y)
	grad=np.zeros( ((columns),1) )
	dataset=dataset[:,:-1]

	for i in range(number_iter):
		y_predicted=predict(weights,dataset)
		for j in range(columns): # we exclude the last as it has the dependent variable
			grad[j]=np.matmul(np.transpose(y_predicted-y),dataset[:,j]) + 2*lmbda*weights[j] # ycap-y multiplied by x column and accounting for the regularization term
			weights[j]-=learn_rate*grad[j] # we make use of the older values of w to get the new ones

	return weights

def lasso(dataset,lmbda):
	rows=dataset.shape[0]
	columns=dataset.shape[1]
	one=np.ones((rows,1))
	dataset=np.hstack((one,dataset))
	learn_rate,number_iter,weights=hardCodeGradientDescent(columns)
	y=dataset[:,-1][np.newaxis] # get the last column containing the power consumption data
	y=np.transpose(y)
	grad=np.zeros( ((columns),1) )
	dataset=dataset[:,:-1]

	for i in range(number_iter):
		y_predicted=predict(weights,dataset)
		for j in range(columns): # we exclude the last as it has the dependent variable
			if(weights[j]<0):
				grad[j]=np.matmul(np.transpose(y_predicted-y),dataset[:,j]) - lmbda # ycap-y multiplied by x column
			else:
				grad[j]=np.matmul(np.transpose(y_predicted-y),dataset[:,j]) + lmbda

			weights[j]-=learn_rate*grad[j] # we make use of the older values of w to get the new ones

	return weights

def normalize(dataset):
	means=dataset.mean(axis=0,keepdims=True)
	stddevs=dataset.std(axis=0,keepdims=True)
	dataset=dataset - means # subtract the corresponding means
	dataset=dataset / stddevs # devide the corresponding standard deviations
	return dataset

def error(weights,dataset):
	rows=dataset.shape[0]
	one=np.ones((rows,1))
	dataset=np.hstack((one,dataset))
	columns=dataset.shape[1]
	y=dataset[:,-1][np.newaxis] # get the last column containing the power consumption data
	y=np.transpose(y)
	dataset=dataset[:,:-1]
	y_predicted=predict(weights,dataset)
	values=(y_predicted-y)
	error = np.matmul(np.transpose(values),values)

	error /=2*rows
	return error

def main():
	limit=10
	dataset = getData()
	rows=dataset.shape[0]
	number_training=hardCodeTraining(rows)

	# splitting to two parts
	training_data=dataset.iloc[:number_training][:]
	test_data=dataset.iloc[number_training: ][:]

	# print (training_data)
	# print (test_data)

	# converting to 2D matrix for ease of use

	training_data=training_data.values
	test_data=test_data.values

	normalised_training_data=normalize(training_data)
	normalised_test_data=normalize(test_data)

	print("***********MLE********************************")
	weights=MLE(normalised_training_data)
	print(weights)
	print("The error is:")
	print(error(weights,normalised_test_data))
	print("***********Gradient Descent********************************")
	weights=gradDes(normalised_training_data)
	print(weights)
	print("The error is:")
	print(error(weights,normalised_test_data))

	initalVal=1
	lmbda = initalVal
	errorList1=np.array([])
	errorList2=np.array([])
	while lmbda>2**(-1*limit):
		print("\n")
		print("Value of lambda %s"%(lmbda))
		print("***********Lasso********************************")
		weights=lasso(normalised_training_data,lmbda)
		print(weights)
		print("The error is:")
		errorList1=np.append(errorList1,error(weights,normalised_test_data))
		print(errorList1[-1])
		print("***********Ridge********************************")
		weights=ridge(normalised_training_data,lmbda)
		print(weights)
		print("The error is:")
		errorList2=np.append(errorList2,error(weights,normalised_test_data))
		print(errorList2[-1])
		print("*************************************************")
		lmbda/=2


	lmbdaVals=[2**(-1*i) for i in range(0, limit)]

	plot1,=plt.plot(lmbdaVals,errorList1)
	plot2,=plt.plot(lmbdaVals,errorList2)
	plt.xlabel('Lambda Value')
	plt.ylabel('Mean Squared Error')
	plt.legend([plot1,plot2],['Lasso Regression','Ridge Regression'])
	plt.show()

	lowerLimit=10**(-4)
	upperLimit=10**(-2)
	xvals=np.array([])
	yvals1=np.array([])
	yvals2=np.array([])

	for i in range(0,101,1):
		i=lowerLimit+lowerLimit*i
		print("\n")
		print("Value of lambda %s"%(i))
		print("***********Lasso********************************")
		weights=lasso(normalised_training_data,i)
		print(weights)
		print("The error is:")
		yvals1=np.append(yvals1,error(weights,normalised_test_data))
		print(yvals1[-1])
		print("***********Ridge********************************")
		weights=ridge(normalised_training_data,i)
		print(weights)
		print("The error is:")
		yvals2=np.append(yvals2,error(weights,normalised_test_data))
		print(yvals2[-1])
		print("*************************************************")
		xvals=np.append(xvals,i)

	plot1,=plt.plot(xvals,yvals1)
	plot2,=plt.plot(xvals,yvals2)
	plt.xlabel('Lambda Value')
	plt.ylabel('Mean Squared Error')
	plt.legend([plot1,plot2],['Lasso Regression','Ridge Regression'])
	plt.show()

if __name__=='__main__':
	main()
