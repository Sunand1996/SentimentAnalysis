import nltk

posFileName = "C:\\Users\sunan\\PycharmProjects\\rt-polaritydata\\rt-polarity.pos"
negFileName = "C:\\Users\sunan\\PycharmProjects\\rt-polaritydata\\rt-polarity.neg"
with open(posFileName,'r') as f:
    positiveReviews = f.readlines()
with open(negFileName,'r') as f:
    negativeReviews = f.readlines()

'''
Training and data from the reviews
'''
testSplitIndex = 2500

testNegativeReviews = negativeReviews[testSplitIndex+1:]
testPositiveReviews = positiveReviews[testSplitIndex+1:]

trainingNegativeReviews = negativeReviews[:testSplitIndex]
trainingPositiveReviews = positiveReviews[:testSplitIndex]

'''
Vocab of training data
'''
def getVocabulary(self):
    positiveWordList = [word for line in trainingPositiveReviews for word in line.split()]
    negativeWordList = [word for line in trainingNegativeReviews for word in line.split()]
    allWordList = [item for sublist in [positiveWordList,negativeWordList] for item in sublist]
    allWordSet = list(set(allWordList))
    vocabulary = allWordSet
    return vocabulary

'''
Extract features - Create word tuples
'''
def extract_features(self,review):
    review_words = set(review)
    features = {}
    for word in self.vocabulary:
        features[word] = (word in review_words)
    return features

'''
Setting up training data:
A list of tuple: First element in each tuple is the review, secoond element is the label
Ex: (amazing, positive)
(worst, negative)
'''
def getTrainingData(self):
    negTaggedTrainingReview = [{'review':oneReview.split(), 'label':'negative'} for oneReview in trainingNegativeReviews]
    posTaggedTrainingReview = [{'review':oneReview.split(), 'label':'positive'} for oneReview in trainingPositiveReviews]
    fullTaggedTrainingData = [item for sublist in [negTaggedTrainingReview,posTaggedTrainingReview] for item in sublist]
    trainingData = [(review['review'], review['label']) for review in fullTaggedTrainingData]
    return trainingData

'''
Feature Extraction: 
Input training data and a function object, output is in correct feature vector form 
Bayes classifier needs only feature vector and labels to be trained 

Ex:
Feature Vector (1,0,0,0) (Amazing)     Label - Positive
                (0,1,1,1,0,0,0) (Worst movie ever)  Label - Negative 
'''
'''
Training the classifier:
Input feature vector and label, output is ready to use classifier
'''
def getTrainedNaiveBayesClassifier(self,extract_features, trainingData):
    trainingFeatures = nltk.classify.apply_features(extract_features,trainingData)
    trainedClassifer = nltk.NaiveBayesClassifier.train(trainingFeatures)
    return trainedClassifer

def naiveBayesSentiCalc(self,review):
    problemInstance = review.split()
    problemFeatures = extract_features(problemInstance)
    return self.trainedClassifer.classify(problemFeatures)

'''
Use the classifier to the test data:
Invoke using a test harness
'''
def getTestReviewSenti(self,naiveBayesSentiCalc):
    testNegResults = [naiveBayesSentiCalc(review) for review in testNegativeReviews]
    testPosResults = [naiveBayesSentiCalc(review) for review in testPositiveReviews]
    label = {'positive':1,'negative':-1}
    numericNegResults = [label[x] for x in testNegResults]
    numericPosResults = [label[x] for x in testPosResults]
    return {'positive':numericPosResults, 'negative':numericNegResults}

def runDiagnostics(reviewResult):
    positiveReviewsResult = reviewResult['positive result']
    negativeReviewsResult = reviewResult['negative result']
    true_positive = float(sum(x>0 for x in positiveReviewsResult))/len(positiveReviewsResult)
    true_negative = float(sum(x<0 for x in negativeReviewsResult))/len(negativeReviewsResult)
    totalAccurate = float(sum(x>0 for x in positiveReviewsResult))+ float(sum(x<0 for x in negativeReviewsResult))
    total = len(positiveReviewsResult) + len(negativeReviewsResult)
    print("Accuracy positive="+"%0.2f" % (true_positive*100)+"%")
    print("Accuracy negative=" + "%0.2f" % (true_negative * 100) + "%")
    print("Overall Accuracy =" + "%0.2f" % (totalAccurate * 100/total) + "%")

runDiagnostics(getTestReviewSenti(naiveBayesSentiCalc))
