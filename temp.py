__author__ = 'Erin'

import csv
import pandas

sub = csv.reader(open('acc_tmp/submission_version_1.1.csv','rt'),delimiter=' ')
x=[]
y=[]
z=[]
for i in sub:
    x.append(i[0])
    y.append(i[1])
    j=i[0].split('_')
    if int(j[1])<=4:
        z.append(0)
    else:
        z.append(1)


output = pandas.DataFrame(data={"id":x, "sentiment":y,"answer":z})
output.to_csv("acc_tmp/submission_version.csv",index=False,quoting=3)
