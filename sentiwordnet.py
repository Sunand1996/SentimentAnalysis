import  nltk
from nltk.corpus import sentiwordnet as swn
from string import punctuation
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english')+ list(punctuation))
posFileName = "C:\\Users\sunan\\PycharmProjects\\rt-polaritydata\\rt-polarity.pos"
negFileName = "C:\\Users\sunan\\PycharmProjects\\rt-polaritydata\\rt-polarity.neg"
with open(posFileName,'r') as f:
    positiveReviews = f.readlines()
with open(negFileName,'r') as f:
    negativeReviews = f.readlines()

def superNaiveSentiment(review):
    reviewPolarity = 0.0
    numExceptions = 0
    for word in review.lower().split():
        numMeaning = 0
        if word in stopwords:
            continue
        weight = 0.0
        try:
            for meaning in swn.senti_synets(word):
                if meaning.pos_score()> meaning.neg_score():
                    weight = weight +(meaning.pos_score() - meaning.neg_score())
                    numMeaning = numMeaning+1
                elif meaning.pos_score() < meaning.neg_score():
                    weight = weight - (meaning.neg_score() - meaning.pos_score())
                    numMeaning = numMeaning+1
        except:
            numExceptions = numExceptions+1
        if numMeaning>0:
            reviewPolarity = reviewPolarity + (weight/numMeaning)
    return reviewPolarity

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

runDiagnostics(getReview(superNaiveSentiment))

