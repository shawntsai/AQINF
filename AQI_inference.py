
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
import time 
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext
start_time = time.time()

sc = SparkContext(appName="AQI_inference")

numPartitions = 2
rawData = sc.textFile('label_features2.csv', numPartitions)
# rawData = sc.textFile('testingLabel3.csv', numPartitions)
header = rawData.first()
rawData = rawData.filter(lambda x: x != header)
numPoints = rawData.count()
sampleData = rawData.take(5)
print 'sample label data: '
print sampleData

uRawData = sc.textFile('unlabeled_feature.csv', numPartitions)
uheader = uRawData.first()
uRawData = uRawData.filter(lambda x:x != uheader)
numUpoints = uRawData.count()
sampleUpoints = uRawData.take(5)
print 'sample unlabel feature data: '
print sampleUpoints

from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.regression import LabeledPoint

nor = Normalizer()
parseUDataInit = uRawData.map(lambda up: DenseVector([float(t) for t in up.split(',')]))
# normalization l2

numFeature = len(parseUDataInit.first())
featureMax = []
for feature in range(numFeature):
	featureMax.append(parseUDataInit.map(lambda up: up[feature]).max())

parseUDataInit = parseUDataInit.map(lambda up: up / featureMax)
print 'after 0-1', parseUDataInit.first()

parseUDataInit = nor.transform(parseUDataInit)
print 'first unlabel after normalize: ', parseUDataInit.first()

def parsePoint(line):
    """Converts a comma separated unicode string into a `LabeledPoint`.

    Args:
        line (unicode): Comma separated unicode string where the first element is the label and the
            remaining elements are features.

    Returns:
        LabeledPoint: The line is converted into a `LabeledPoint`, which consists of a label and
            features.
    """
    tokens = line.split(',')
    label, features = tokens[0], tokens[1:]
    return LabeledPoint(label, features)

parseDataInit = rawData.map(parsePoint)
# parse and do feature normalization l2
parseDataInit = parseDataInit.map(lambda lp: LabeledPoint(lp.label, lp.features))
featureMax = []
for feature in range(numFeature):
	featureMax.append(parseDataInit.map(lambda lp: lp.features[feature]).max())
parseDataInit = parseDataInit.map(lambda lp: LabeledPoint(lp.label, lp.features / featureMax))
print 'after 0-1: ', parseDataInit.first()

parseDataInit = parseDataInit.map(lambda lp: LabeledPoint(lp.label, nor.transform(lp.features)))
print 'first label after norm : ', parseDataInit.first()

onlyLabels = parseDataInit.map(lambda point: point.label).collect()
minLabel = min(onlyLabels)
maxLabel = max(onlyLabels)
print 'min AQI', minLabel, 'maxAQI', maxLabel




"""
	plot the feature and the label relation

"""

# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# dataValues = parseDataInit.map(lambda lp: lp.features.toArray()).collect()
# # print dataValues
# dataValuesU = parseUDataInit.map(lambda lp: lp.toArray()).take(100)
# # print dataValuesU

# def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
#                 gridWidth=1.0):
#     """Template for generating the plot layout."""
#     plt.close()
#     fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
#     ax.axes.tick_params(labelcolor='#999999', labelsize='10')
#     for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
#         axis.set_ticks_position('none')
#         axis.set_ticks(ticks)
#         axis.label.set_color('#999999')
#         if hideLabels: axis.set_ticklabels([])
#     plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
#     map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
#     return fig, ax

# # generate layout and plot
# fig, ax = preparePlot(np.arange(.5, 11, 1), np.arange(.5, 49, 1), figsize=(8,7), hideLabels=True,
#                       gridColor='#eeeeee', gridWidth=1.1)
# image = plt.imshow(dataValuesU,interpolation='nearest', aspect='auto', cmap=cm.Greys)
# for x, y, s in zip(np.arange(-.125, numFeature, 1), np.repeat(-.75, numFeature), [str(x) for x in range(numFeature)]):
#     plt.text(x, y, s, color='#999999', size='10')
# plt.text(4.7, -3, 'Feature', color='#999999', size='11'), ax.set_ylabel('Observation')
# plt.show()
# pass





"""
	normalize feature
	make all the features <= 1
	column by column of feature
	collect to list of columns
"""
featureList = []
for feature in range(numFeature):
	# no normalization
	featureList.append(parseDataInit.map(lambda lp: lp.features[feature]).collect())

	# -1 ~ 1
	# featureMax = parseDataInit.map(lambda lp: lp.features[feature]).max()
	# normalizeFeature = parseDataInit.map(lambda lp: lp.features[feature] / float(featureMax)).collect()
	# featureList.append(normalizeFeature)
	
featureList_u = []
for feature in range(numFeature):
	# no normalization
	featureList_u.append(parseUDataInit.map(lambda up: up[feature]).collect())

	# -1 ~ 1
	# featureMax = parseUDataInit.map(lambda up: up[feature]).max()
	# normalizeFeature = parseUDataInit.map(lambda up: up[feature] / float(featureMax)).collect()
	# featureList_u.append(normalizeFeature)


import math
# def doRegression(labeled_data, step):
# 	"""
# 	Args: labels  step = alpha = 1/((numPoints * math.sqrt(numberOfIter))) 
# 	# output: regression model
# 	"""
# 	numIters = 100
# 	useIntercept = True  # intercept
# 	miniBatchFraction = 1.0 #miniBatchFraction
# 	# numFeature = 1
# 	# initialWeights = np.array([1.0]*numFeature)
# 	# reg = 1e-1
# 	# regType = 'l2'
# 	lrm = LinearRegressionWithSGD.train(labeled_data,
# 										step =step,
# 		 								iterations=numIters,
# 		 								miniBatchFraction= miniBatchFraction,
# 		 								initialWeights= None, # or None
# 		 								# regParam= reg,
# 		 								# regType=regType,
# 		 								intercept=useIntercept
# 		 								)

# 	return lrm

# from pyspark.sql.types import *
# customSchema = StructType([ 
#     StructField("AQI", DoubleType(), True), 
#     StructField("x", DoubleType(), True), 
#     StructField("y", DoubleType(), True), 
#     # StructField("PM10", DoubleType(), True), 
#     # StructField("PM2.5", DoubleType(), True),
# 	StructField("hw_len", DoubleType(), True),
# 	StructField("rd_len", DoubleType(), True),
# 	StructField("num_intersection", DoubleType(), True),
#     ])

# customSchema2 = StructType([ 
#     StructField("x", DoubleType(), True), 
#     StructField("y", DoubleType(), True), 
#     # StructField("PM10", DoubleType(), True), 
#     # StructField("PM2.5", DoubleType(), True),
# 	StructField("hw_len", DoubleType(), True),
# 	StructField("rd_len", DoubleType(), True),
# 	StructField("num_intersection", DoubleType(), True),
#     ])

# sqlContext = SQLContext(sc)
# df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('label_features.csv')
# df_u = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('unlabeled_feature.csv')
# df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('label_features2.csv',schema = customSchema)
# df_u = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('unlabeled_feature.csv',schema = customSchema2)

# print tree structure
# df.printSchema()

# numFeature = len(df_u.columns)
# print 'number of features: ', numFeature
# print 'sample labels : '
# sampleLabel = df.take(5)
# print sampleLabel
# print 'sample un Labels: '
# sampleUnLabel = df_u.take(5)
# print sampleUnLabel
# print names of list
# print df.columns

# df.select('x').show()
# df.filter(df['x'] > 5).show()


from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

def diffMatrixInit(featureList):
	"""Calculate the differnece of the node's feature fk = ||fk(u) - fk(v)||
	Args: two different numpy array of same length(1 X N) or same numpy array
	Returns: N X N numpy array
	"""
	TfeatureList = featureList.reshape(len(featureList), 1)
	return np.absolute(TfeatureList - featureList) 
	# return (featureList1 - featureList2)

def differencePoint(feature):
	"""Calculate the node's feature difference delta fk = ||fk(u) - fk(v)||
	Args: feature list

	Returns:
		the absolute feature difference of Triangular matrix in flat vector
	"""
	feature = np.array(feature)
	featureMat = diffMatrixInit(feature)
	number_of_nodes = len(featureMat[0])
	offset_of_diagonal = 1
	flatMat = featureMat[np.triu_indices(number_of_nodes, offset_of_diagonal)]
	# flatMat = normalize(flatMat)
	featureDiff_RDD = sc.parallelize(flatMat)
	return featureDiff_RDD

def calcRMSE(labelsAndPreds):
    """Calculates the root mean squared error for an `RDD` of (label, prediction) tuples.

    Args:
        labelsAndPred (RDD of (float, float)): An `RDD` consisting of (label, prediction) tuples.

    Returns:
        float: The square root of the mean of the squared errors.
    """
    return np.sqrt(labelsAndPreds
                   .map(lambda (label, prediction):(label - prediction) ** 2)
                   .mean())


from scipy import stats



labelDiff_RDD = differencePoint(onlyLabels)
# print 'labelDiff_RDD'
# print labelDiff_RDD.collect()
# min_AQIDiff = min(labelDiff_RDD.collect())
# print 'minimum AQI difference: ', min_AQIDiff

weightAndIntercept = []

for feature in range(numFeature):	
	featureDiff_RDD = differencePoint(featureList[feature])
	# print 'feature difference list'
	# print featureDiff_RDD.collect()
	labelPoint = (labelDiff_RDD
				 .zip(featureDiff_RDD)
				 .map(lambda (labelDiff, featureDiff): LabeledPoint(labelDiff, [featureDiff])))
	# print labelPoint.take(100)
	
	# numberOfIter = 200
	# alpha = 1/((labelPoint.count() * math.sqrt(numberOfIter))) #step
	# alpha = 1
	# regressionTime = time.time()
	# lrm = doRegression(labelPoint, alpha)
	# print ("---Spark regression time %s seconds ---"% (time.time() - regressionTime))
	# weightAndIntercept.append([lrm.weights[0], lrm.intercept]) 
	# print 'from spark'
	# print weightAndIntercept[feature]
	# samplePoint = labelPoint.take(2)[1]
	# print 'testing sample point--------------------------------'
	# print samplePoint
	# print lrm.predict(samplePoint.features)
	# labelsAndPreds = labelPoint.map(lambda lp: (lp.label, lrm.predict(lp.features)))
	# rmse = calcRMSE(labelsAndPreds)
	# print 'calculate RMSE :', rmse
	print 'from scipy'
	regressionTime = time.time()
	regressResult = stats.linregress(np.array(featureDiff_RDD.collect()), np.array(labelDiff_RDD.collect()))
	print ("---scipy regression time %s seconds ---"% (time.time() - regressionTime))
	print 'slope and intercept'
	slope, intercept = regressResult[0], regressResult[1]
	print slope, intercept
	weightAndIntercept.append([slope,intercept]) 
	labelsAndPreds = labelPoint.map(lambda lp: (lp.label, lp.features[0] * slope + intercept))
	rmse = calcRMSE(labelsAndPreds)
	print 'calculate RMSE :', rmse

print featureList
print weightAndIntercept

nodeList = []

# PI of feature
# Initialize feature weight PI(k) = 1, where k = 1,2 ..., m.
featureWeightList = [1.0] * numFeature


"""
Wu,v: sumWeightAffinity
PI: featureWeight 

Construct affinity graph AG.

"""

sumWeightAffinity_RDD = sc.parallelize([])
print 'number of labels: ', numPoints
print 'number of unlabels: ', numUpoints
AFGraph_RDD = []
for feature in range(numFeature):
	# nodeList.append(featureList_u[feature] + featureList[feature])
	nodeList.append(np.append(featureList_u[feature], featureList[feature]))
	slope = weightAndIntercept[feature][0]
	intercept = weightAndIntercept[feature][1]
	AF = (differencePoint(nodeList[feature])
			.map(lambda diff: diff * slope + intercept).cache())
	# normalizeAF
	# maxAF = AF.max()
	# AF = AF.map(lambda obj: obj/maxAF)	

	# print 'feature: ', feature + 1
	# print AF.collect()
	AFGraph_RDD.append(AF)
	if feature == 0: #initialize sumWeightAffinity_RDD
		sumWeightAffinity_RDD = AFGraph_RDD[feature]
	else:
		sumWeightAffinity_RDD = (sumWeightAffinity_RDD
								.zip(AFGraph_RDD[feature])
								.map(lambda (x ,y): x + y))
# maxSumWeightAf = sumWeightAffinity_RDD.max()
# sumWeightAffinity_RDD = (sumWeightAffinity_RDD
# 						.map(lambda p: math.exp(-p / maxSumWeightAf))
# 						.cache())

sumWeightAffinity_RDD = (sumWeightAffinity_RDD
						.map(lambda p: math.exp(-p))
						.cache())

# print sumWeightAffinity_RDD.collect()

# """

# Normalize edge Weight here

# """
# maxSumWeightAf = sumWeightAffinity_RDD.max()
# sumWeightAffinity_RDD = (sumWeightAffinity_RDD
# 						.map(lambda obj: obj/ maxSumWeightAf)
# 						.cache())

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
# Wu_uv_RDD = weightMatrix_RDD.zipWithIndex().filter(lambda (row, index): index < numUpoints).keys()


Wu_uv_RDD = sc.parallelize(weightMatrix[:numUpoints])
Wuu_RDD = Wu_uv_RDD.map(lambda row: row[:numUpoints])
Wuv_RDD = Wu_uv_RDD.map(lambda row: row[numUpoints:])
# print 'Wuu'
# print Wuu_RDD.collect()
# print 'Wuv'
# print Wuv_RDD.collect()
# print len(Wuu_RDD.collect())

# sum row of matrix
Duu_RDD = Wu_uv_RDD.map(lambda lp: lp.sum())
# print Duu_RDD.collect()
Duu = np.diag(np.array(Duu_RDD.collect()))
# print 'Duu'
# print Duu
# print len(Duu)
Wuu = Wuu_RDD.collect()
toInverse = Duu - Wuu
# print toInverse
from numpy.linalg import inv
inverse = inv(toInverse)
# print 'inverse'
# print inverse
# print inverse.shape


"""
initialize Pv

"""
labelRange = 100
probabilityDist = np.array([[0.0] * labelRange] * int(numPoints))

"""
input indexlist(list) , probabilityList (matrix numpy 2D array)
elements number of two inputs should be same
output probability Distribution(based on normal distribution)
"""

# def insertDistribution(indexInsertList, probabilityDist):
# 	for i in range(len(indexInsertList)):
# 		probabilityDist[i][int(indexInsertList[i])] = 1.0
# 	return probabilityDist
from scipy.stats import norm
variance = 3
def insertDistribution(indexInsertList, probabilityDist):
	for i in range(len(indexInsertList)):
		for j in range(len(probabilityDist[i])):
			probability = norm(int(indexInsertList[i]), variance).pdf(j)
			probabilityDist[i][j] = probability
	return probabilityDist

Pv = insertDistribution(onlyLabels, probabilityDist)
np.set_printoptions(threshold = 50000)

# print 'pv distribution'
# print Pv


# dot in numpy is matrix multiplication m x n  multiply n x p  =  m x p
# u*u * u*v * v*x
Wuv = np.array(Wuv_RDD.collect())
# print Wuv.shape
Pu = np.dot(inverse, Wuv).dot(Pv)
# print Pu.shape

# print 'before loop'
# print np.dot(inverse, Wuv)

qu = sc.parallelize(Pu).map(lambda row: np.argmax(row)).collect()
print qu

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
flatPu_RDD = sc.parallelize(flatPu)
entropy = flatPu_RDD.map(lambda lp: entropyCalculate(lp)).sum() * (-1.0) / numUpoints
# print entropy


# AFGraph_RDD -> AFgraph  4s -> 1s

def updateFeatureWeightList(edgeWeight, AFGraph, featureWeightList):
	"""update PIk, PIk = (1 - 2 Wu,v * Af(delta fk(u, v))) * PIk

	Args: Wuv (edgeWeight) 
		  Af(delta fk(u, v)) (AFGraph) list of RDD(each feature one RDD)
		  PIk (featureWeightList) numpy array of feature weight
	returns: PIk
	"""
	sumProportion = 0.0
	proportion = []
	# edgeWeight = edgeWeight_RDD.collect()
	for feature in range(numFeature):
		# proportion.append(AFGraph_RDD[feature].zip(edgeWeight_RDD).map(lambda (x, y): x * y).sum() * 2)
		proportion.append(np.sum(np.dot(edgeWeight, AFGraph[feature]) * 2))
		sumProportion +=  proportion[feature]

	"""
		need to do further work here
	"""
	sumProportion *= 10
	for feature in range(numFeature):
		featureWeightList[feature] = (1 - proportion[feature] / float(sumProportion)) * featureWeightList[feature]
	return featureWeightList

def calculateEdgeWeight(AFGraph_RDD, featureWeightList):
	"""update Wuv = exp(-Sum(PIk^2 * Af(delta fk(u, v))))

	Args: 
		  Af(delta fk(u, v)) (AFGraph_RDD) list of RDD(each feature one RDD)
		  PIk (featureWeightList) numpy array of feature weight
	returns: Wuv of RDD 
	"""
	sumWeightAffinity_RDD = sc.parallelize([])
	for feature in range(numFeature):
		featureWeight = featureWeightList[feature] * featureWeightList[feature]
		tempAFGraph_RDD = (AFGraph_RDD[feature]
									   .map(lambda AF: featureWeight * AF))
		#initialize sumWeightAffinity_RDD
		if feature == 0: 
			sumWeightAffinity_RDD = tempAFGraph_RDD
		else:
			sumWeightAffinity_RDD = (sumWeightAffinity_RDD
									.zip(tempAFGraph_RDD)
									.map(lambda (x ,y): x + y))
	# maxSumWeightAf = sumWeightAffinity_RDD.max()
	# sumWeightAffinity_RDD = sumWeightAffinity_RDD.map(lambda p: math.exp(-p / float(maxSumWeightAf)))
	sumWeightAffinity_RDD = sumWeightAffinity_RDD.map(lambda p: math.exp(-p)).cache()
	# print 'Edge weights'
	# print sumWeightAffinity_RDD.take(40)

	# """

	# Normalize edge Weight here

	# """
	# maxSumWeightAf = sumWeightAffinity_RDD.max()
	# sumWeightAffinity_RDD = (sumWeightAffinity_RDD
	# 						.map(lambda obj: obj/ maxSumWeightAf)
	# 						.cache())

	return sumWeightAffinity_RDD


# print sumWeightAffinity_RDD.collect()

# print AFGraph_RDD[0].collect()

# print featureWeightList

# print weightedAFGraph_RDD[0].collect()

# np.set_printoptions(threshold = 'nan')
np.set_printoptions(threshold = 50000)
edgeWeight = sumWeightAffinity_RDD.collect()
AFGraph = []
for feature in range(numFeature):
	AFGraph.append(AFGraph_RDD[feature].collect())

# print Pv
entropyDiff = 1.0
count = 0
while (entropyDiff > 0.01 and count < 100):
	loopTime = time.time()
	count = count + 1
	print count 			
	featureWeightTime = time.time()
	featureWeightList = updateFeatureWeightList(edgeWeight, AFGraph, featureWeightList)
	print ("---feature weight Run Time %s seconds ---"% (time.time() - featureWeightTime)) # 1.03045010567
	print featureWeightList

	calculateEdgeWeightTime = time.time()
	edgeWeight_RDD = calculateEdgeWeight(AFGraph_RDD, featureWeightList)
	print ("---calculateEdgeWeight Time %s seconds ---"% (time.time() - calculateEdgeWeightTime))# 0.169363021851
	getMatrixTime = time.time()	
	edgeWeight = edgeWeight_RDD.collect()
	weightMatrix = squareform(np.array(edgeWeight))
	# weightMatrix_RDD = sc.parallelize(weightMatrix)
	# Wu_uv_RDD = weightMatrix_RDD.zipWithIndex().filter(lambda (row, index): index < numUpoints).keys()
	
	# Wu_uv_RDD = sc.parallelize(weightMatrix[:numUpoints])
	# Wuu_RDD = Wu_uv_RDD.map(lambda row: row[:numUpoints])
	# Wuv_RDD = Wu_uv_RDD.map(lambda row: row[numUpoints:])
	# Duu_RDD = Wu_uv_RDD.map(lambda lp: lp.sum())
	# Duu = np.diag(np.array(Duu_RDD.collect()))
	# Wuu = np.array(Wuu_RDD.collect()) 
	# Wuv = np.array(Wuv_RDD.collect())
	# print 'Wuv'
	# print Wuv
	
	
	Wu_uv = weightMatrix[:numUpoints]
	Wuu = Wu_uv[:, :numUpoints]
	Wuv = Wu_uv[:, numUpoints:]
	Duu = np.diag(np.sum(Wu_uv, axis =1)) # sum of each rows
	print ("---get Wuu, Wuv, Duu Run Time %s seconds ---"% (time.time() - getMatrixTime))
	
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
	InverseTime = time.time()	
	Pu = inverse.dot(Wuv).dot(Pv)
	print ("---product Run Time %s seconds ---"% (time.time() - InverseTime))

	# print Pu
	flatPu = Pu.reshape(-1)
	flatPu_RDD = sc.parallelize(flatPu)
	caculateEntropyTime = time.time()
	newEntropy = flatPu_RDD.map(lambda lp: entropyCalculate(lp)).sum()*(-1.0) / numUpoints
	print ("---Entropy Run Time %s seconds ---"% (time.time() - caculateEntropyTime))
	# print np.dot(inverse, Wuv)

	qu = sc.parallelize(Pu).map(lambda row: np.argmax(row)).collect()
	print qu

	entropyDiff = math.fabs(newEntropy - entropy)
	print entropyDiff
	entropy = newEntropy 
	print ("---TotalLOOP Time %s seconds ---"% (time.time() - loopTime))


qu = sc.parallelize(Pu).map(lambda row: np.argmax(row)).collect()
print qu
print ("---TotalRun Time %s seconds ---"% (time.time() - start_time))


sc.stop()


