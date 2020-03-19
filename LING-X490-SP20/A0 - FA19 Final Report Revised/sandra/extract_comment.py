import re

#with open("tessi", "r") as IN:
with open("train.csv", "r") as IN:
    line = IN.readline().strip()
    line = IN.readline().strip()
    while line:
        c = line.find(",")
        l1 = line[c+1:]
        c = l1.find(",")
        targets = l1[:c]
        targetc = float(targets)
        l2 = l1[c+1:]
        if len(l2) > 0:
            if re.match('\"', l2): # i.e. this example has either a comma or is multi-line
                done = 0
                l2 = l2[1:]
                while not done and not re.search('([^"]|[^"]"")",', l2): # ", should end the example, but "", is part of the example ...
                    nl = IN.readline().strip()
                    if re.search('([^"]|[^"]"")",', nl):
                        mo = re.search('(^.*)([^"]|"")",', nl)
                        if mo:
                            n = mo.groups()
                            l2 += " " + n[0] + n[1]
                            done = 1
                        else:
                            print "XXX hmmmm"
                    else:
                        l2 += " " + nl
                if not done:
                    mo = re.search('(^.*)([^"]|[^"]"")",', l2)
                    if mo:
                        n = mo.groups()
                        l2 = n[0] + n[1]
                    else:
                        print "XXX hmmmm"
  
                print targetc, "\t", l2
            else:
                c = l2.find(',')
                print targetc, "\t", l2[:c]
        else:
            print "XXX error"

        line = IN.readline().strip()
