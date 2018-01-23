# always start like this
from guineapig import *
import sys
import math

# supporting routines can go here


def tokens(curDoc):
    for label in curDoc[0].split(','):
        for word in curDoc[1].split():
            yield '%s,%s' % (label, word)
            yield '%s,*' % label


def getPredProb(row):
    (replies, (v1, v2, v3, v4)) = row
    docid, word, labelWordCount = replies
    dummy1, labelCount = v1
    dummy2, nlabel = v2
    dummy3, labelAnyCount = v3
    dummy4, nwords = v4
    word_prob = {}
    label_prob = {}
    for label, count in labelCount.items():
         label_prob[label] = math.log((count+0.0) / (nlabel))
    for label in labelCount.keys():
        word_prob[label]=math.log((labelWordCount.get(label,0)+1.0)/(labelAnyCount[label]+nwords))
    return (docid, label_prob, word_prob)


def storeLabelCount(accum, label):
    accum[label] = accum.get(label, 0) + 1
    return accum


def storeLabelAnyCount(accum, labelAnyCount):
    dummy, labelAnyCount = labelAnyCount
    for (labelAny, count) in labelAnyCount:
        accum[labelAny] = count
    return accum

def storeLabelWordCount(accum,labelWordCount):
    labelWord,count=labelWordCount
    label=labelWord.split(',')[0]
    accum[label]=count
    return accum

def getNumLabels(labelCount):
    total = 0
    for(labelAny, count)in labelCount.items():
        total += count
    return total


def flattenLabelCount(labels):
    for label in labels:
        yield label

def getTotalProb(accum,row):
    docid, label_prob, word_prob=row
    for label in label_prob.keys():
        accum[label]=accum.get(label,label_prob[label])+word_prob[label]
    return accum

def pred(row):
    docid, prob=row
    k=list(prob.keys())
    v=list(prob.values())
    bestProb=max(v)
    bestPred=k[v.index(bestProb)]
    return(docid, bestPred, bestProb)



# always subclass Planner, definition of a view


class NB(Planner):
    # params is a dictionary of params given on the command line.
    # e.g. trainFile = params['trainFile']coim
    params = GPig.getArgvParams()
    curDoc = ReadLines(params['trainFile']) | \
        Map(by=lambda line: line.strip().split('\t')[1:])
    # curDoc = ReadLines('abstract.smaller.train') | \
    #     Map(by=lambda line: line.strip().split('\t')[1:])

    # same as the counter c in hw1
    labelWordCount = Flatten(curDoc, by=tokens) | Group(by=lambda x: x, reducingTo=ReduceToCount())
    # reorganize to c'
    word_labelWordCount = Group(labelWordCount, by=lambda(labelWord, count): labelWord.split(',')[1],
                                        reducingTo=ReduceTo(dict, by=lambda accum,labelWordCount:storeLabelWordCount(accum,labelWordCount)))


    # produce word-docid request
    curTest = ReadLines(params['testFile']) | Map(by=lambda line: line.strip().split('\t'))
    # curTest = ReadLines('abstract.smaller.test') | Map(by=lambda line: line.strip().split('\t'))
    idWords = Map(curTest, by=lambda(docid, labels, doc): (docid, doc.split()))
    requests = FlatMap(idWords, by=lambda(docid, words): map(lambda w: (w, docid), words))

    # produce docid-word-labelWordCount reply (docid,word,{label:count})
    replies = Join(Jin(requests, by=lambda(word, docid): word),
                   Jin(word_labelWordCount, by=lambda(word, labelWordCount): word)) |\
              ReplaceEach(by=lambda((w1, docid),(w2, labelWordCount)): (docid, w1, labelWordCount)) 

    '''support Views
       labelCount: ('labelCount', {label:count})
       nlabel: ('nlabel',count)
       labelAny: ('label,*',{'label,*':count})
       nwords: ('nwords',count)
    '''

    labelCount = Map(curDoc, by=lambda(labels, doc): labels.split(',')) | \
        Flatten(by=flattenLabelCount) | \
        Group(by=lambda label: 'labelCount',
              reducingTo=ReduceTo(dict, by=lambda accum, label: storeLabelCount(accum, label)))

    nlabel = Map(labelCount, by=lambda (dummy, labelCount): ('nlabel', getNumLabels(labelCount)))
    labelAnyCount = Filter(word_labelWordCount, by=lambda(word, labelWordCount): word == '*') 

    nword = Filter(word_labelWordCount,
                   by=lambda(word, labelWordCount): word != '*') | Distinct() | \
        Group(by=lambda row: 'nword', reducingTo=ReduceToCount())

    # # combined the request and reply (docid,[(word_1,[('label_1,word_1',count),...]),(word_2,[...]),...])
    # combined = Group(replies, by=lambda(docid, w, labelWordCount): docid,
    #                  retaining=lambda(docid, w, labelWordCount): (w, labelWordCount))

    # augment information needed
    # (docid,word,{label:count}),(('labelCount', {label:count}),('nlabel',count),('label,*',{'label,*':count}),('nwords',count))
    info = Augment(replies, sideviews=[labelCount, nlabel, labelAnyCount, nword],
                   loadedBy=lambda v1, v2, v3, v4: (GPig.onlyRowOf(v1), GPig.onlyRowOf(v2), GPig.onlyRowOf(v3), GPig.onlyRowOf(v4)))

    output = Map(info, by=lambda row: getPredProb(row)) |\
              Group(by = lambda(docid, label_prob, word_prob):docid, 
             reducingTo= ReduceTo(dict, by=lambda accum,row:getTotalProb(accum,row)))|\
              Map(by=lambda row: pred(row))


# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)
