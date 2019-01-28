# ******************************
# @Time    : 1/26/19 1:32 PM
# @Author  : Maggie Fang
# @Software: PyCharm
# @Version : Python3 
# ******************************

# init spark context
from pyspark import SparkContext
from operator import *
from pyspark.mllib.recommendation import *

sc = SparkContext("local","Music Recommender")

# Data load and preprocess
artistData= sc.textFile("./data_raw/artist_data_small.txt").map(lambda x: x.split("\t")).map(lambda x: (int(x[0]),x[1]))
artistAlias= sc.textFile("./data_raw/artist_alias_small.txt").map(lambda x: x.split("\t")).map(lambda x: (int(x[0]),int(x[1])))
userArtistData= sc.textFile("./data_raw/user_artist_data_small.txt").map(lambda x: x.split(" ")).map(lambda x: (int(x[0]),int(x[1]),int(x[2])))
artistAlias = artistAlias.collectAsMap()

# function to change the wrong artist id in userArtistData to the right one
def reviseUserArtist(data):
    id = data[1]
    for key,val in artistAlias.items():
        if key == id:
            id = val
            break
    return (data[0],id,data[2])
userArtistData = userArtistData.map(reviseUserArtist)


# Data Exploration

# group the user_artist data by userId and calculate the sum of songs of the users listening history and the times
playSum = userArtistData.map(lambda x:(x[0],x[2])).groupByKey().map(lambda x: (x[0],sum(x[1]),len(x[1]))).collect()
result = sorted(playSum,key = lambda x:x[1],reverse=True)[0:3]
for i in range(3):
    print("User " + str(result[i][0]) + " has a total play count of  " + str(result[i][1]) + " and a mean play count of "+ str(int(result[i][1]/result[i][2])) +".")

# split into trainData,validationData,testData and cache
trainData, validationData, testData = userArtistData.randomSplit([0.4, 0.4, 0.2], 13)
trainData.cache()
validationData.cache()
testData.cache()
print(trainData.take(3))
print(validationData.take(3))
print(testData.take(3))
print(trainData.count())
print(validationData.count())
print(testData.count())


def modelEval(model, dataset):
    allArtist = list(set(userArtistData.map(lambda x: x[1]).collect()))   # all artist list
    validUser = list(set(testData.map(lambda x: x[0]).collect()))         # users to validate

    validationDict  = dataset.map(lambda x :(x[0],x[1])).groupByKey().mapValues(list).collectAsMap()
    trainDataDict = trainData.map(lambda x :(x[0],x[1])).groupByKey().mapValues(list).collectAsMap()

    totalScore = 0.0
    for user in validUser:
        trueArtist = validationDict[user]    # the true listening artist list for validating user
        nonTrainArtist = set(allArtist) - set(trainDataDict[user])   # the non-train artist for this user
        tmpList = []
        for artist in nonTrainArtist:
            tmpList.append((user,artist))

        predict = model.predictAll(sc.parallelize(tmpList))
        res = predict.takeOrdered(len(trueArtist), key=lambda x: -x[2]) # get the highest predictresult with the len of truetrueArtist

        predictArtist = []
        for item in res:
            predictArtist.append(item[1])  # predict artist list for this user
        overlap = set(trueArtist) & set(predictArtist)

        totalScore +=len(overlap)/len(trueArtist)

    return totalScore/len(validUser)  # the average score

# Model train
ranks = [2, 10, 20]
for rank in ranks:
    model = ALS.trainImplicit(trainData, rank = rank, seed=345)
    score = modelEval(model, validationData)
    print('The model score for rank ' +str(rank)+ ' is ' + str(score))

bestModel = ALS.trainImplicit(trainData, rank=10, seed=345)
print(modelEval(bestModel, testData))

# recommend top 5 artist for specific user
top5ListRating = bestModel.recommendProducts(1059637,5)
artistData = artistData.collectAsMap();
i = 0
for recommend in top5ListRating:
    for id,name in artistData.items():
        if(id == recommend[1]):
            print("Artist "+str(i)+": "+name)
            i+=1






