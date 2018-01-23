#!/usr/bin/env python
import re
import sys


def tokenizeDoc(cur_doc):
    return re.findall('\\w+', cur_doc)


def incrSpl(c, e, bs):
    if (e not in c) and (len(c) == bs):
        for (k, v) in c.items():
            print('%s\t%d' % (k, v))
        c = {}
    c[e] = c.get(e, 0) + 1


counters = {}
bufferSize = 1000000000
v = set()

for cur_doc in sys.stdin:
    docID, labels, doc = cur_doc.split('\t')
    labels = labels.split(',')
    for label in labels:
        ey = 'Y=%s' % label
        eany = 'Y=*'
        incrSpl(counters, ey, bufferSize)
        incrSpl(counters, eany, bufferSize)
        features = tokenizeDoc(doc)
        for word in features:
            ewordy = 'Y=%s,W=%s' % (label, word)
            eanyy = 'Y=%s,W=*' % label
            incrSpl(counters, ewordy, bufferSize)
            incrSpl(counters, eanyy, bufferSize)
            v.add(word)
for (k, v) in counters.items():
    print('%s\t%d' % (k, v))
print(len(v))
