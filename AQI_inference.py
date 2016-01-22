
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
import time 
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import DenseVector
start_time = time.time()

sc = SparkContext(appName="sparkMatrixTest")
sqlContext = SQLContext(sc)

# df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('label_features.csv')
# df_u = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('unlabeled_feature.csv')

# assume I have four features, 3 points

rows = sc.parallelize([[1, 2, 3], [4, 5, 6],
                       [7, 8, 9], [10, 11, 12]])



# input: labels 
# output: regression outputs
import math
def doRegression(labeled_data, step):
	numIters = 100
	useIntercept = True  # intercept
	miniBatchFraction = 1.0 #miniBatchFraction
	# numberOfFeature = 1
	# initialWeights = np.array([1.0]*numberOfFeature)
	# reg = 1e-1
	# regType = 'l2'
	lrm = LinearRegressionWithSGD.train(labeled_data,
										step =step,
		 								iterations=numIters,
		 								miniBatchFraction= miniBatchFraction,
		 								initialWeights= None, # or None
		 								# regParam= reg,
		 								# regType=regType,
		 								intercept=useIntercept
		 								)

	return lrm
# input DenseVector
# >>> v = lin.Vectors.dense([6,7,8,9,10])

# def commonDiffMatrixInit(featureList1, featureList2):
#     featureList2 = featureList2.reshape(-1, 1);
#     return abs(featureList2 - featureList1)



def commonDiffMatrixInit(featureList1,featureList2):
    featureList1 = featureList1.reshape(-1, 1);
    return (featureList1 - featureList2)

# feature1 = DenseVector([-1.0, -4.0, -4.0, -5.0, -7.0, -7.0, -9.0, -9.0, -10.0, -10.0, -10.0, -11.0, -13.0, -19.0, -21.0, -21.0, -3.0, -3.0, -4.0, -6.0, -6.0, -8.0, -8.0, -9.0, -9.0, -9.0, -10.0, -12.0, -18.0, -20.0, -20.0, 0.0, -1.0, -3.0, -3.0, -5.0, -5.0, -6.0, -6.0, -6.0, -7.0, -9.0, -15.0, -17.0, -17.0, -1.0, -3.0, -3.0, -5.0, -5.0, -6.0, -6.0, -6.0, -7.0, -9.0, -15.0, -17.0, -17.0, -2.0, -2.0, -4.0, -4.0, -5.0, -5.0, -5.0, -6.0, -8.0, -14.0, -16.0, -16.0, 0.0, -2.0, -2.0, -3.0, -3.0, -3.0, -4.0, -6.0, -12.0, -14.0, -14.0, -2.0, -2.0, -3.0, -3.0, -3.0, -4.0, -6.0, -12.0, -14.0, -14.0, 0.0, -1.0, -1.0, -1.0, -2.0, -4.0, -10.0, -12.0, -12.0, -1.0, -1.0, -1.0, -2.0, -4.0, -10.0, -12.0, -12.0, 0.0, 0.0, -1.0, -3.0, -9.0, -11.0, -11.0, 0.0, -1.0, -3.0, -9.0, -11.0, -11.0, -1.0, -3.0, -9.0, -11.0, -11.0, -2.0, -8.0, -10.0, -10.0, -6.0, -8.0, -8.0, -2.0, -2.0, 0.0])
# featureMat = commonDiffMatrixInit(feature1,feature1)
# # flatting the triangular matrix 
# offset =1
# feature1 = featureMat[np.triu_indices(numberOfData,offset)]

# AQI = DenseVector([4.0, -1.0, -7.0, -18.0, -1.0, 4.0, -12.0, 14.0, 9.0, -16.0, 10.0, -2.0, 8.0, 17.0, 33.0, 6.0, -5.0, -11.0, -22.0, -5.0, 0.0, -16.0, 10.0, 5.0, -20.0, 6.0, -6.0, 4.0, 13.0, 29.0, 2.0, -6.0, -17.0, 0.0, 5.0, -11.0, 15.0, 10.0, -15.0, 11.0, -1.0, 9.0, 18.0, 34.0, 7.0, -11.0, 6.0, 11.0, -5.0, 21.0, 16.0, -9.0, 17.0, 5.0, 15.0, 24.0, 40.0, 13.0, 17.0, 22.0, 6.0, 32.0, 27.0, 2.0, 28.0, 16.0, 26.0, 35.0, 51.0, 24.0, 5.0, -11.0, 15.0, 10.0, -15.0, 11.0, -1.0, 9.0, 18.0, 34.0, 7.0, -16.0, 10.0, 5.0, -20.0, 6.0, -6.0, 4.0, 13.0, 29.0, 2.0, 26.0, 21.0, -4.0, 22.0, 10.0, 20.0, 29.0, 45.0, 18.0, -5.0, -30.0, -4.0, -16.0, -6.0, 3.0, 19.0, -8.0, -25.0, 1.0, -11.0, -1.0, 8.0, 24.0, -3.0, 26.0, 14.0, 24.0, 33.0, 49.0, 22.0, -12.0, -2.0, 7.0, 23.0, -4.0, 10.0, 19.0, 35.0, 8.0, 9.0, 25.0, -2.0, 16.0, -11.0, -27.0])

# AQIMat = commonDiffMatrixInit(AQI,AQI)
# AQI = AQIMat[np.triu_indices(numberOfData,offset)]

# AQIRDD = sc.parallelize(AQI)
# AQIfeature1RDD = sc.parallelize(feature1)

# from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

# Label_feature_RDD = AQIRDD.zip(AQIfeature1RDD)


# labelPointRDD = Label_feature_RDD.map(lambda (lable, feature):LabeledPoint(lable, [feature]))
# lrm = doRegression(labelPointRDD)
# Weight_intercept = (lrm.weights[0], lrm.intercept)
# print Weight_intercept
# from scipy import stats

# regressResult = stats.linregress(np.array(feature1), np.array(AQI))
# print 'slope and intercept'
# print regressResult[0], regressResult[1]

# print labelPoints.collect()
# print AQIRDD.count()
# print AQIfeature1RDD.count()
# print Label_feature_RDD.count()






# l_feature = DenseVector([1,2,3,4,5])
# u_feature = DenseVector([2,4,6,8,10])

# tempMatrix = np.vstack((  
# 			np.hstack([commonDiffMatrixInit(l_feature,l_feature),
# 					  commonDiffMatrixInit(u_feature,l_feature)]),
# 			np.hstack([commonDiffMatrixInit(l_feature,u_feature),
# 					  commonDiffMatrixInit(u_feature,u_feature)])))
# print tempMatrix
# numberOfData = 10
# offset = 0
# tempMatrixTriAngular = tempMatrix[np.triu_indices(numberOfData,offset)]

# print tempMatrixTriAngular

# TriRDD = sc.parallelize(tempMatrixTriAngular)
# Affinity = TriRDD.map(lambda lp: lp*interceptAndWeight[1] + interceptAndWeight[0])
from pyspark.sql.types import *
customSchema = StructType([ 
    StructField("AQI", DoubleType(), True), 
    StructField("x", DoubleType(), True), 
    StructField("y", DoubleType(), True), 
    # StructField("PM10", DoubleType(), True), 
    # StructField("PM2.5", DoubleType(), True),
	StructField("hw_len", DoubleType(), True),
	StructField("rd_len", DoubleType(), True),
	StructField("num_intersection", DoubleType(), True),
    ])

customSchema2 = StructType([ 
    StructField("x", DoubleType(), True), 
    StructField("y", DoubleType(), True), 
    # StructField("PM10", DoubleType(), True), 
    # StructField("PM2.5", DoubleType(), True),
	StructField("hw_len", DoubleType(), True),
	StructField("rd_len", DoubleType(), True),
	StructField("num_intersection", DoubleType(), True),
    ])

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('label_features2.csv',schema = customSchema)
df_u = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('unlabeled_feature.csv',schema = customSchema2)

# print tree structure
df.printSchema()

# print names of list
# print df.columns

# df.select('x').show()
# df.filter(df['x'] > 5).show()





from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel


# input: feature list
# output: flatten difference of point in Triangular matrix (RDD format)

def differencePoint_feature(feature):
	feature = DenseVector(feature)
	featureMat = commonDiffMatrixInit(feature, feature)
	number_of_nodes = len(featureMat[0])
	offset_of_diagonal = 1
	featureDiff_RDD = sc.parallelize(featureMat[np.triu_indices(number_of_nodes, offset_of_diagonal)])
	return featureDiff_RDD

def calcRMSE(labelsAndPreds):
    """Calculates the root mean squared error for an `RDD` of (label, prediction) tuples.

    Args:
        labelsAndPred (RDD of (float, float)): An `RDD` consisting of (label, prediction) tuples.

    Returns:
        float: The square root of the mean of the squared errors.
    """
    return np.sqrt(labelsAndPreds
                   .map(lambda (label, prediction):(label - prediction)**2)
                   .mean())



numberOfFeature = 5

from scipy import stats
AQI_RDD = df.map(lambda row: row.AQI)
AQI = AQI_RDD.collect()
AQIDiff_RDD = differencePoint_feature(AQI)
min_AQIDiff = min(AQIDiff_RDD.collect())
print 'minimum AQI difference', min_AQIDiff
# print 'AQIDiff list'
# print AQIDiff_RDD.collect()
featureList = []
weightAndIntercept = []
# for feature in range(numberOfFeature):

# 	#how many data we train 
# 	# print nTrain
# 	# nTrain = labeled_data.count()
# 	# normalize column 3 and 4 need to fix it at the beginning later~
# 	if feature == 2 or feature == 3: 
# 		featureList.append(df.map(lambda row: row[feature + 1]/1000).collect())
# 	else:
# 		featureList.append(df.map(lambda row: row[feature + 1]).collect())
# 	featureDiff_RDD = differencePoint_feature(featureList[feature])
# 	# print 'feature difference list'
# 	# print featureDiff_RDD.collect()
# 	label_feature_RDD = AQIDiff_RDD.zip(featureDiff_RDD)
# 	labelPoint = label_feature_RDD.map(lambda (AQIDiff, featureDiff): LabeledPoint(AQIDiff,[featureDiff]))
# 	# print labelPoint.collect()
# 	# print labelPoint.take(100)
# 	numberOfIter = 100
# 	alpha = 1/((labelPoint.count()*math.sqrt(numberOfIter))) #step
# 	# if feature == 4:
# 	# 	alpha = 0.0000001
# 	# elif feature ==5:
# 	# 	alpha = 0.000000001
# 	lrm = doRegression(labelPoint, alpha)
# 	weightAndIntercept.append([lrm.weights[0],lrm.intercept]) 
# 	print 'from spark'
# 	print weightAndIntercept[feature]
	
# 	# samplePoint = labelPoint.take(2)[1]
# 	# print 'testing sample point--------------------------------'
# 	# print samplePoint
# 	# print lrm.predict(samplePoint.features)
# 	labelsAndPreds = labelPoint.map(lambda lp: (lp.label, lrm.predict(lp.features)))
# 	rmse = calcRMSE(labelsAndPreds)
# 	print 'calculate RMSE :', rmse
# 	print 'from scipy'
# 	regressResult = stats.linregress(np.array(featureList[feature]), np.array(AQI))
# 	# print 'slope and intercept'
# 	# print regressResult[0], regressResult[1]
# 	labelsAndPreds = labelPoint.map(lambda lp: (lp.label, lp.features[0]*regressResult[0]+ regressResult[1]))
# 	print 'slope and intercept'
# 	rmse = calcRMSE(labelsAndPreds)
# 	print 'calculate RMSE :', rmse


weightAndIntercept = [[-0.69729318740583912, 1.0494948836315574], [-0.14770461351845288, 1.060398896989658], [0.0033279794393285601, 1.038432462955263], [0.48115789048484564, 1.0244840501954933], [0.11618042024424484, 1.0168396362253236]]
featureList = [[8.0, 9.0, 12.0, 12.0, 13.0, 15.0, 15.0, 17.0, 17.0, 18.0, 18.0, 18.0, 19.0, 21.0, 27.0, 29.0, 29.0], [23.0, 14.0, 15.0, 21.0, 22.0, 12.0, 20.0, 20.0, 22.0, 17.0, 18.0, 33.0, 6.0, 20.0, 14.0, 22.0, 38.0], [0.0, 0.0, 0.0, 0.7373505697, 0.0, 0.0, 0.0, 0.0, 3.292903302, 5.73481938, 4.938099391, 3.636624576, 0.0, 0.0, 0.0, 0.0, 0.0], [26.12956149, 18.520306100000003, 49.14497907, 41.33075339, 50.18863679, 24.750776849999998, 32.52901084, 46.15369164, 58.12550086, 29.66263824, 43.13642085, 22.07082922, 29.887151030000002, 10.27214524, 12.807370290000001, 6.4005287410000005, 20.90531046], [24.0, 26.0, 148.0, 154.0, 107.0, 29.0, 50.0, 128.0, 151.0, 51.0, 84.0, 28.0, 56.0, 6.0, 30.0, 3.0, 51.0]]
# print featureList
# print weightAndIntercept

featureList_u = []
nodeList = []

# PI of feature
# Initialize feature weight PI(k) = 1, where k = 1,2 ..., m.
featureWeightList = [1]*numberOfFeature



"""
Wu,v: sumWeightAffinity
PI: featureWeight 

Construct affinity graph AG.

"""

def weightUpdate(featureWeight, affinity):
	return featureWeight * featureWeight * affinity


sumWeightAffinity_RDD = sc.parallelize([])
numberOfLabelPoint = df.count() # 17
numberOfnonLabelPoint = df_u.count() # 1122
print numberOfLabelPoint
print numberOfnonLabelPoint
AFGraph_RDD = []
weightedAFGraph_RDD = []
for feature in range(numberOfFeature):
	if feature == 2 or feature == 3:
		featureList_u.append(df_u.map(lambda row: row[feature]/1000.0).collect())
	else:
	 	featureList_u.append(df_u.map(lambda row: row[feature]).collect())
	# print featureList_u[feature]
	nodeList.append(featureList_u[feature] + featureList[feature])

	slope = weightAndIntercept[feature][0]
	intercept = weightAndIntercept[feature][1]
	AF = differencePoint_feature(nodeList[feature]).map(lambda diff: diff * slope + intercept).cache()
	AFGraph_RDD.append(AF)
	weightedAF = AFGraph_RDD[feature].map(lambda AF: weightUpdate(featureWeightList[feature], AF)).cache()
	weightedAFGraph_RDD.append(weightedAF)
	if feature == 0: #initialize sumWeightAffinity_RDD
		sumWeightAffinity_RDD = weightedAFGraph_RDD[feature]
	else:
		sumWeightAffinity_RDD = sumWeightAffinity_RDD.zip(weightedAFGraph_RDD[feature]).map(lambda (x ,y): x + y)

# maxSumWeightAf = sumWeightAffinity_RDD.max()
# sumWeightAffinity_RDD = sumWeightAffinity_RDD.map(lambda p: math.exp(-p / maxSumWeightAf)).cache()
sumWeightAffinity_RDD = sumWeightAffinity_RDD.map(lambda p: math.exp(-p)).cache()

# print sumWeightAffinity_RDD.collect()



# converst array to matrix

# 1 2 3 4 5 6

# 0 1 2 3
# 1 0 4 5
# 2 4 0 6
# 3 5 6 0

from scipy.spatial.distance import squareform
weightMatrix = squareform(np.array(sumWeightAffinity_RDD.collect()))
# print weightMatrix
# weightMatrix_RDD = sc.parallelize(weightMatrix)
# Wu_uv_RDD = weightMatrix_RDD.zipWithIndex().filter(lambda (row, index): index < numberOfnonLabelPoint).keys()
Wu_uv_RDD = sc.parallelize(weightMatrix[:numberOfnonLabelPoint])
Wuu_RDD = Wu_uv_RDD.map(lambda row: row[:numberOfnonLabelPoint])
Wuv_RDD = Wu_uv_RDD.map(lambda row: row[numberOfnonLabelPoint:])
# print Wuu_RDD.collect()
# print Wuv_RDD.collect()
# print len(Wuu_RDD.collect())

# sum row of matrix
Duu_RDD = Wu_uv_RDD.map(lambda lp: lp.sum())
# print Duu_RDD.collect()
Duu = np.diag(np.array(Duu_RDD.collect()))
# print len(Duu)
Wuu = Wuu_RDD.collect()
toInverse = Duu - Wuu
# print toInverse
from numpy.linalg import inv
inverse = inv(toInverse)
# print inverse.shape


"""
initialize Pv
"""
maxAQI = int(AQI_RDD.max())
print 'max AQI: ', maxAQI
probabilityDist = np.array([[0.0] * (maxAQI + 1)] * numberOfLabelPoint)

"""
input indexlist(list) , probabilityList (matrix numpy 2D array)
elements number of two inputs should be same
output probability Distribution
"""

# def insertDistribution(indexInsertList, probabilityDist):
# 	for i in range(len(indexInsertList)):
# 		probabilityDist[i][int(indexInsertList[i])] = 1.0
# 	return probabilityDist
from scipy.stats import norm
variance = 5
minimumProbability = 0.00001
def insertDistribution(indexInsertList, probabilityDist):
	for i in range(len(indexInsertList)):
		for j in range(len(probabilityDist[i])):
			probability = norm(int(indexInsertList[i]), variance).pdf(j) * 100
			if probability > minimumProbability:
				probabilityDist[i][j] = probability 
	return probabilityDist

Pv = insertDistribution(AQI, probabilityDist)
np.set_printoptions(threshold = 'nan')
# print 'pv distribution'
# print Pv


# dot in numpy is matrix multiplication m x n  multiply n x p  =  m x p
# u*u * u*v * v*x
Wuv = np.array(Wuv_RDD.collect())
# print Wuv.shape
Pu = np.dot(inverse, Wuv).dot(Pv)
# print Pu.shape

import sys

def entropyCalculate(entity):
    # if entity == 0.0 or entity == 1.0:
    #     return 0.0
    # else:
    entropy = 0.0
    try:
    	entropy = entity * math.log(entity, 2) + (1 - entity) * math.log(1 - entity, 2)
    except (ZeroDivisionError, ValueError):
    	pass
    return entropy
flatPu = Pu.reshape(-1)

# count =0
# for elem in flatPu:
# 	if elem > 1 or elem < 0:
# 		print elem
# 	elif elem != 0:
# 		count = count +1
# print flatPu
# print count

flatPu_RDD = sc.parallelize(flatPu)
entropy = flatPu_RDD.map(lambda lp: entropyCalculate(lp)).sum() * (-1.0) / numberOfnonLabelPoint
# print entropy


def updateSumWeightAffinity(weightedAFGraph_RDD, sumWeightAffinity_RDD, featureWeightList):
	sumWeightAffinity_RDD = sc.parallelize([])
	# numberOfFeature = 5
	for feature in range(numberOfFeature):
		weightedAFGraph_RDD[feature] = weightedAFGraph_RDD[feature].map(lambda AF: weightUpdate(featureWeightList[feature], AF))
		#initialize sumWeightAffinity_RDD
		if feature == 0: 
			sumWeightAffinity_RDD = weightedAFGraph_RDD[feature]
		else:
			sumWeightAffinity_RDD = sumWeightAffinity_RDD.zip(weightedAFGraph_RDD[feature]).map(lambda (x ,y): x + y)
	# maxSumWeightAf = sumWeightAffinity_RDD.max()
	# sumWeightAffinity_RDD = sumWeightAffinity_RDD.map(lambda p: math.exp(-p / maxSumWeightAf))
	sumWeightAffinity_RDD = sumWeightAffinity_RDD.map(lambda p: math.exp(-p))
	return sumWeightAffinity_RDD



"""
IMPORTANT : normalization 
(1 -|2Wuv * Af| )PI


"""
# def updateFeatureWeightList(sumWeightAffinity_RDD, AFGraph_RDD, featureWeightList):
# 	# numberOfFeature = 5
# 	for feature in range(numberOfFeature):
# 		proportion = AFGraph_RDD[feature].zip(sumWeightAffinity_RDD).map(lambda (x, y): 2 * x * y).sum()
# 		featureWeightList[feature] = (1 - proportion) * featureWeightList[feature]
# 	return featureWeightList

# def updateFeatureWeightList(sumWeightAffinity_RDD, AFGraph_RDD, featureWeightList):
# 	maxProportion = 0.0
# 	proportion = []
# 	for feature in range(numberOfFeature):
# 		proportion.append(AFGraph_RDD[feature].zip(sumWeightAffinity_RDD).map(lambda (x, y): 2 * x * y).sum())
# 		if maxProportion < proportion[feature]:
# 			maxProportion =  proportion[feature]
# 	for feature in range(numberOfFeature):
# 		featureWeightList[feature] = (1 - math.fabs(proportion[feature]/maxProportion)) * featureWeightList[feature]
# 	return featureWeightList
def updateFeatureWeightList(sumWeightAffinity_RDD, AFGraph_RDD, featureWeightList):
	sumProportion = 0.0
	proportion = []
	sumWeightAff = sumWeightAffinity_RDD.collect()
	for feature in range(numberOfFeature):
		# proportion.append(AFGraph_RDD[feature].zip(sumWeightAffinity_RDD).map(lambda (x, y): x * y).sum() * 2)
		proportion.append(np.sum(np.dot(AFGraph_RDD[feature].collect(), sumWeightAff)) * 2)
		sumProportion +=  proportion[feature]	
	for feature in range(numberOfFeature):
		featureWeightList[feature] = (1 - proportion[feature] / sumProportion) * featureWeightList[feature]
	return featureWeightList

# from operator import add
# print 'do algorithm I'
# print '1'
# print sumWeightAffinity_RDD.collect()
# print '2'
# print AFGraph_RDD[0].collect()
# print '3'
# print featureWeightList
# print '4'
# print weightedAFGraph_RDD[0].collect()

# np.set_printoptions(threshold = 'nan')
np.set_printoptions(threshold = 5000)

# print Pv
entropyDiff = 1.0
count = 0
while (entropyDiff > 0.00001):
	loopTime = time.time()
	count = count + 1
	print count 			
	featureWeightTime = time.time()
	featureWeightList = updateFeatureWeightList(sumWeightAffinity_RDD, AFGraph_RDD, featureWeightList)
	print ("---feature weight Run Time %s seconds ---"% (time.time() - featureWeightTime)) # 41.6238129139->10.3459608555
	print featureWeightList
	updateSumWeightAffinityTime = time.time()
	sumWeightAffinity_RDD = updateSumWeightAffinity(weightedAFGraph_RDD, sumWeightAffinity_RDD, featureWeightList)
	print ("---updateSumWeightAffinity Time %s seconds ---"% (time.time() - updateSumWeightAffinityTime))# 7.58352303505
	weightMatrix = squareform(np.array(sumWeightAffinity_RDD.collect()))
	# weightMatrix_RDD = sc.parallelize(weightMatrix)
	# Wu_uv_RDD = weightMatrix_RDD.zipWithIndex().filter(lambda (row, index): index < numberOfnonLabelPoint).keys()
	Wu_uv_RDD = sc.parallelize(weightMatrix[:numberOfnonLabelPoint])
	Wuu_RDD = Wu_uv_RDD.map(lambda row: row[:numberOfnonLabelPoint])
	Wuv_RDD = Wu_uv_RDD.map(lambda row: row[numberOfnonLabelPoint:])
	Duu_RDD = Wu_uv_RDD.map(lambda lp: lp.sum())
	Duu = np.diag(np.array(Duu_RDD.collect()))
	Wuu = np.array(Wuu_RDD.collect()) # each node will turn into 1.0
	# print 'Wuu'
	# print Wuu
	# print 'Duu'
	# print Duu
	toInverse = Duu - Wuu # each node will turn into same [[-1]*u]*v as loop 
	# print 'Wuu'
	# print Wuu
	# print 'toinverse'
	# print toInverse
	InverseTime = time.time()	
	inverse = inv(toInverse)
	print ("---inverse Run Time %s seconds ---"% (time.time() - InverseTime))# 2.95696997643
	
	# print 'Inverse'
	# print inverse
	Wuv = np.array(Wuv_RDD.collect())
	# print 'Wuv'
	# print Wuv
	Pu = inverse.dot(Wuv).dot(Pv)
	
	# print Pu
	flatPu = Pu.reshape(-1)
	flatPu_RDD = sc.parallelize(flatPu)
	caculateEntropyTime = time.time()
	newEntropy = flatPu_RDD.map(lambda lp: entropyCalculate(lp)).sum()*(-1.0) / numberOfnonLabelPoint
	print ("---Entropy Run Time %s seconds ---"% (time.time() - caculateEntropyTime))
	
	entropyDiff = math.fabs(newEntropy - entropy)
	print entropyDiff
	entropy = newEntropy 
	print ("---TotalLOOP Time %s seconds ---"% (time.time() - loopTime))


qu = sc.parallelize(Pu).map(lambda row: np.argmax(row)).collect()
print qu


print ("---TotalRun Time %s seconds ---"% (time.time() - start_time))




# print 'here is featureMat'
# feature_x = DenseVector([xRDD.collect()])
# featureMat = commonDiffMatrixInit(feature_x,feature_x)
# print featureMat
# # flatting the triangular matrix 
# feature1 = featureMat[np.triu_indices(numberOfData,offset)]

















# from pyspark.mllib.linalg import Matrices
# from pyspark.mllib.linalg.distributed import BlockMatrix

# feature1 = DenseVector([1,2,3,4,5,6,7,8])
# featureMat = commonDiffMatrixInit(feature1,feature1)
# # flatting the triangular matrix 
# feature1 = featureMat[np.triu_indices(numberOfData,offset)]

# AQI = DenseVector([20,30,40,50,60,70,80,90])

# AQIMat = commonDiffMatrixInit(AQI,AQI)
# AQI = AQIMat[np.triu_indices(numberOfData,offset)]

# AQIRDD = sc.parallelize(AQI)
# AQIfeature1RDD = sc.parallelize(feature1)

# from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

# Label_feature_RDD = AQIRDD.zip(AQIfeature1RDD)

# labelPointRDD = Label_feature_RDD.map(lambda (lable, feature):LabeledPoint(lable, [feature]))
# interceptAndWeight = doRegression(labelPointRDD)
 























# # Create an RDD of sub-matrix blocks.
# blocks = sc.parallelize([((0, 0), Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])), 
#                          ((1, 0), Matrices.dense(3, 2, [7, 8, 9, 10, 11, 12]))])

# # Create a BlockMatrix from an RDD of sub-matrix blocks.
# mat = BlockMatrix(blocks, 3, 2)

# print mat.toLocalMatrix()

# # Get its size.
# m = mat.numRows() # 6
# n = mat.numCols() # 2


# # Get the blocks as an RDD of sub-matrix blocks.
# blocksRDD = mat.blocks

# # Convert to a LocalMatrix.
# localMat = mat.toLocalMatrix()

# # Convert to an IndexedRowMatrix.
# indexedRowMat = mat.toIndexedRowMatrix()

# # Convert to a CoordinateMatrix.
# coordinateMat = mat.toCoordinateMatrix()


# dm1 = Matrices.dense(2, 3, [1, 2, 3, 4, 5, 6])
# dm2 = Matrices.dense(2, 3, [7, 8, 9, 10, 11, 12])
# dm3 = Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])
# dm4 = Matrices.dense(3, 2, [7, 8, 9, 10, 11, 12])
# sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 1, 2], [7, 11, 12])
# blocks1 = sc.parallelize([((0, 0), dm1), ((0, 1), dm2)])
# blocks2 = sc.parallelize([((0, 0), dm3), ((1, 0), dm4)])
# blocks3 = sc.parallelize([((0, 0), sm), ((1, 0), dm4)])
# mat1 = BlockMatrix(blocks1, 2, 3)
# mat2 = BlockMatrix(blocks2, 3, 2)
# mat3 = BlockMatrix(blocks3, 3, 2)

# print mat1.toLocalMatrix()
# print mat2.toLocalMatrix()
# print mat1.multiply(mat2).toLocalMatrix()




sc.stop()


















# def aqiPosition(insertedPoints):
# 	for point in insertedPoints:
# 		yield point

