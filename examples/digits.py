
"""
Digits
===========================

Work in progress
"""


import emlearn

import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, datasets
import serial.tools.list_ports as port_list

rnd = 11
digits = datasets.load_digits()
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=rnd)

print('Loading digits dataset. 8x8=64 features')

# 0.95+ with n_estimators=40, max_depth=20
# 0.90+ with n_estimators=10, max_depth=10
trees = 40
max_depth = 20
print('Training {} trees with max_depth {}'.format(trees, max_depth))
model = RandomForestClassifier(n_estimators=trees, max_depth=max_depth, random_state=rnd)
model.fit(Xtrain, ytrain)

# Predict
ypred = model.predict(Xtest)
print('Accuracy on validation set {:.2f}%'.format(metrics.accuracy_score(ypred, ytest)*100))

m = numpy.max(Xtrain), numpy.min(Xtrain)

filename = 'digits.h'
cmodel = emlearn.convert(model)
code = cmodel.save(file=filename)

print('Wrote C code to', filename)

assert len(port_list.comports()) > 0, "No serial ports available"

port = port_list.comports()[0].device # grab the first serial port
print('Classify on microcontroller via', port)
import serial
device = serial.Serial(port=port, baudrate=115200, timeout=0.1) 

repetitions = 10
Y_pred = []
times = []
for idx,row in enumerate(Xtest):
   # send
   values = [idx, repetitions] + list(row)
   send = ';'.join("{}".format(v) for v in values) + '\n'
   device.write(send.encode('ascii'))
   resp = device.readline()

   # receive
   tok = resp.decode('ascii').strip().split(';')
   retvals = [int(v) for v in tok]
   (request,micros,prediction,reps),values = retvals[:4], retvals[4:]

   assert request == idx
   assert reps == repetitions
   err = numpy.array(values) - row
   assert numpy.sum(err) == 0, err

   Y_pred.append(prediction)
   times.append(micros / 1000)
   #print(idx, prediction, reps, micros)

print('Confusion matrix')
print(metrics.confusion_matrix(Y_pred, ytest))

avg = numpy.mean(times) / repetitions
stddev = numpy.std(numpy.array(times) / repetitions)
print('Time per classification (ms): {:.2f} avg, {:.2f} stdev'.format(avg, stddev))
