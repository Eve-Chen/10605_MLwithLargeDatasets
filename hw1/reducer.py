#!/usr/bin/env python
import sys


def outputPreviousKey(k, s):
    if k != None:
        print('%s\t%d' % (k, s))


previousKey = None
sumForPreviousKey = 0
for record in sys.stdin:
    event, delta = record.split('\t')
    delta = int(delta.rstrip('\n'))
    if event == previousKey:
        sumForPreviousKey += delta
    else:
        outputPreviousKey(previousKey, sumForPreviousKey)
        previousKey = event
        sumForPreviousKey = delta
outputPreviousKey(previousKey, sumForPreviousKey)
