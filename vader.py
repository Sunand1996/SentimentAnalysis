import  nltk
from nltk.sentiment import vader
sia = vader.SentimentIntensityAnalyzer()

posFileName = "C:\\Users\sunan\\PycharmProjects\\rt-polaritydata\\rt-polarity.pos"
negFileName = "C:\\Users\sunan\\PycharmProjects\\rt-polaritydata\\rt-polarity.neg"
with open(posFileName,'r') as f:
    positiveReviews = f.readlines()
with open(negFileName,'r') as f:
    negativeReviews = f.readlines()

def vaderSentiment(review):
    return sia.polarity_scores(review) ['compound']

def getReview(sentimentCalc):
    negativeResult = [sentimentCalc(oneNegativeReview) for oneNegativeReview in negativeReviews]
    positiveResult = [sentimentCalc(onePositiveReview) for onePositiveReview in positiveReviews]
    return {'positive result':positiveResult, 'negative result':negativeResult}

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

vaderResults = runDiagnostics(getReview(vaderSentiment))
print(vaderResults)





