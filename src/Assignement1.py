# AUTHOR : ADRIEN BRÉAL
# CREATED ON THE 15/11/2019
# ALGORITHM FOR SPAM CLASSIFICATION, THE POSITIVE CLASS IS THE SPAM ONE
# SPECIFIC FOR THE FOLLOWING DATASET "https://archive.ics.uci.edu/ml/datasets/Spambase"

import os # useful to clean the workspace

### USEFUL CONSTANTS ###
TRAINING_SET_SIZE = 900# size of the training set , size of the whole positive classes set is 1873
DATA_SIZE = 57 # size of the features vector
SEP_CHAR = ',' # separator character used for separate features
PATH = os.path.dirname(__file__)+'/../'
### The model contains 57 sublists, representing the possible interval of values for each feature ###
model=[] 
for i in range(DATA_SIZE):
    model.append([0,0])

### This function is used to separate the dataset into 2 subsets; the training set and the validation set ###
def defineSets():

    with open(PATH+"res/spambase_data.data","r") as dataset:
        with open(PATH+"res/trainingset.data","w") as trainingset:
            for i in range(TRAINING_SET_SIZE): 
                data = dataset.readline()
                if not data:
                    break
                data = data.split(SEP_CHAR)
                if "1" in data[-1]:
                    data.pop(-1)
                    trainingset.write(SEP_CHAR.join(data))
                    trainingset.write("\n")
        
        with open(PATH+"res/validationset.data","w") as validationset:
            for i in range(4601-TRAINING_SET_SIZE):
                data = dataset.readline()
                if not data:
                    break
                validationset.write(data)

### This function is called at each start of the program for cleaning the files in the res folder ###               
def cleanWorkSpace():
     try:
        os.remove(PATH+"res/trainingset.data")
        os.remove(PATH+"res/validationset.data")
        os.remove(PATH+"res/result.data")
     except Exception:
        pass

### This function trains the model, consisting in a list of sublist with 2 values defining the interval that a spam mail features should be in ###
def trainModel():
    with open(PATH+"res/trainingset.data","r") as trainingset:
       while 1:
           data = trainingset.readline()
           if not data:
               break
           data = data.split(SEP_CHAR)
           data[-1] = data[-1].rstrip('\n')
           data =[float(x) for x in data]
           for i in range(DATA_SIZE):
              if data[i] > model[i][1]:
                 model[i][1] = data[i]
              if data[i] < model[i][0]:
                 model[i][0] = data[i]
    

### This function tests if a feature is consistent with the model ###
def validate(data):
    for i in range(DATA_SIZE):
        if data[i] >= model[i][0] and data[i] <= model[i][1]:
            continue
        else:
            return False
    return True

### This function tests the whole validation set ###
def checkModel():
    with open(PATH+"res/validationset.data","r") as validationset:
       with open(PATH+"res/result.data","w") as result:
            while 1:
               data = validationset.readline()
               if not data:
                   break
               data = data.split(SEP_CHAR)
               data.pop(-1)
               data =[float(x) for x in data]
               if validate(data):
                   data.append(1)
               else:
                   data.append(0)
               data = [str(x) for x in data]
               result.write(SEP_CHAR.join(data))
               result.write('\n')

### This function is use to analyze results ###
def analyzeResult():
    TP,FP,TN,FN = 0,0,0,0
    with open(PATH+"res/validationset.data","r") as validationset:
        with open(PATH+"res/result.data") as result:
            while 1:
                res = result.readline()
                tra = validationset.readline()
                if not res:
                    break
                res = res.split(SEP_CHAR)
                tra = tra.split(SEP_CHAR)

                if '0' in tra[-1] and '0' in res[-1]:
                    TN+=1
                if '0' in tra[-1] and '1' in res[-1]:
                    FP+=1
                if '1' in tra[-1] and '1' in res[-1]:
                    TP+=1
                if '1' in tra[-1] and '0' in res[-1]:
                    FN+=1
            displayStatistics(TP,FP,TN,FN)

### This function displays all the statistics from the classification ###
def displayStatistics(TP,FP,TN,FN):
    total = TP+TN+FP+FN
    print("##### STATISTICS FROM THE CLASSIFICATION #####\n")
    print("##### SIZE OF THE TRAINING SET : {} #####\n".format(TRAINING_SET_SIZE))
    print("##### CONFUSION MATRIX #####\n")
    print("\t\t\t___REAL___")
    print("\t\t\tP\tN\n")
    print("\t\t|P\t{}\t{}\n\t\t|".format(TP,FP))
    print("EVALUATED AS\t|\n\t\t|")
    print("\t\t|N\t{}\t{}\n".format(FN,TN))
    print("NUMBER OF POSITIVE : {}".format(TP+FP))
    print("NUMBER OF NEGATIVE : {}".format(FN+TN))
    print("TOTAL : {}".format(total))
    print("CLASS RATIO :{}%\n".format(round(((TP+FN)/(TN+FP))*100)))
    print("NUMBER OF TRUE POSITIVE : {}".format(TP))
    print("NUMBER OF FALSE POSITIVE : {}".format(FP))
    print("NUMBER OF TRUE NEGATIVE : {}".format(TN))
    print("NUMBER OF FALSE NEGATIVE : {}\n".format(FN))
    print("ACCURACY : {}%".format(round(((TN+TP)/total)*100)))
    print("TRUE POSITIVE RATE : {}%".format(round((TP/(TP+FN))*100)))
    print("TRUE NEGATIVE RATE : {}%".format(round((TN/(TN+FP))*100)))
    print("PRECISION : {}%\n".format(round((TP/(TP+FP))*100)))
    print("GENERATED MODEL :\n{}".format(model))

### Main ###
def main():
    cleanWorkSpace()
    defineSets()
    trainModel()
    checkModel()
    analyzeResult()        

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
