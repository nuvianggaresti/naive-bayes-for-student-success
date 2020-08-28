import pandas as pd
import csv
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

####################################################################
def load_data(file_name):
    with open(file_name, mode="r") as file:
        data = pd.read_csv(file_name,sep=',',header=0)
    return data

####################################################################
def split_data(data_arg):
    train_set, test_set = train_test_split(data_arg, test_size=0.2, stratify=data_arg[['Grade']])
    return train_set, test_set


####################################################################
# Constant declarations
CLASS = 6
ALPHA = 1
MAX_VALUE = 5

AA = 0
A = 1
B = 2
C = 3
D = 4
F = 5

M2 = 2
M3 = 3
M4 = 4
M5 = 5

SCHOOL = 0
SEX = 1
ADDRESS = 2
FAMSIZE = 3
PSTATUS = 4
MEDU = 5
FEDU = 6
MJOB = 7
FJOB = 8
REASON = 9
GUARDIAN = 10
TRAVELTIME = 11
STUDYTIME = 12
FAILURES = 13
SCHOOLSUP = 14
FAMSUP = 15
PAID = 16
ACTIVITIES = 17
NURSERY = 18
HIGHER = 19
INTERNET = 20
ROMANTIC = 21
FAMREL = 22
FREETIME = 23
GOOUT = 24
DALC = 25
WALC = 26
HEALTH = 27
ABSENCES = 28
GRADE = 29

GP = 0
MS = 1
F_SEX = 0
M_SEX = 1
U = 0
R = 1
LE3 = 0
GT3 = 1
T_PSTATUS = 0
A_PSTATUS = 1
NONE = 0
LOW = 1
MID = 2
HIGH = 3
VERY_HIGH = 4
TEACHER = 0
HEALTH_JOB = 1
SERVICES = 2
AT_HOME = 3
OTHER_JOB = 4
HOME = 0
REPUTATION = 1
COURSE = 2
OTHER_REASON = 3
MOTHER = 0
FATHER = 1
OTHER_GUARDIAN = 2
YES = 1
NO = 0

# This function was written to minimise code repetition when iterating over data set to obtain categorical distribution
def build_distribution(df, tup, arr, idx):
    """Generalised function to build an array of categorical distribution over features given label.
    Returns said array.
    To be used in conjunction with train() function below.

    Keyword arguments:
    df -- pandas dataframe that stores the data set
    tup -- tuple currently being iterated in train() function
    arr -- array that holds the categorical distribution (unsmooth)
    idx -- class index: A+ = 0, A = 1, B = 2, C = 3, D = 4, F = 5
    """
    for c in df.iteritems():
        if c[0] == 'school':
            if tup.school == 'GP':
                arr[SCHOOL][GP][idx] += 1
            else:  # 'MS'
                arr[SCHOOL][MS][idx] += 1
        elif c[0] == 'sex':
            if tup.sex == 'F':
                arr[SEX][F_SEX][idx] += 1
            else:  # 'M'
                arr[SEX][M_SEX][idx] += 1
        elif c[0] == 'address':
            if tup.address == 'U':
                arr[ADDRESS][U][idx] += 1
            else:  # 'R'
                arr[ADDRESS][R][idx] += 1
        elif c[0] == 'famsize':
            if tup.famsize == 'LE3':
                arr[FAMSIZE][LE3][idx] += 1
            else:  # 'GT3'
                arr[FAMSIZE][GT3][idx] += 1
        elif c[0] == 'Pstatus':
            if tup.Pstatus == 'T':
                arr[PSTATUS][T_PSTATUS][idx] += 1
            else:  # 'A'
                arr[PSTATUS][A_PSTATUS][idx] += 1
        elif c[0] == 'Medu':
            if tup.Medu == 'none':
                arr[MEDU][NONE][idx] += 1
            elif tup.Medu == 'low':
                arr[MEDU][LOW][idx] += 1
            elif tup.Medu == 'mid':
                arr[MEDU][MID][idx] += 1
            else:  # 'high'
                arr[MEDU][HIGH][idx] += 1
        elif c[0] == 'Fedu':
            if tup.Fedu == 'none':
                arr[FEDU][NONE][idx] += 1
            elif tup.Fedu == 'low':
                arr[FEDU][LOW][idx] += 1
            elif tup.Fedu == 'mid':
                arr[FEDU][MID][idx] += 1
            else:  # 'high'
                arr[FEDU][HIGH][idx] += 1
        elif c[0] == 'Mjob':
            if tup.Mjob == 'teacher':
                arr[MJOB][TEACHER][idx] += 1
            elif tup.Mjob == 'health':
                arr[MJOB][HEALTH_JOB][idx] += 1
            elif tup.Mjob == 'services':
                arr[MJOB][SERVICES][idx] += 1
            elif tup.Mjob == 'at_home':
                arr[MJOB][AT_HOME][idx] += 1
            else:  # 'other'
                arr[MJOB][OTHER_JOB][idx] += 1
        elif c[0] == 'Fjob':
            if tup.Fjob == 'teacher':
                arr[FJOB][TEACHER][idx] += 1
            elif tup.Fjob == 'health':
                arr[FJOB][HEALTH_JOB][idx] += 1
            elif tup.Fjob == 'services':
                arr[FJOB][SERVICES][idx] += 1
            elif tup.Fjob == 'at_home':
                arr[FJOB][AT_HOME][idx] += 1
            else:  # 'other'
                arr[FJOB][OTHER_JOB][idx] += 1
        elif c[0] == 'reason':
            if tup.reason == 'home':
                arr[REASON][HOME][idx] += 1
            elif tup.reason == 'reputation':
                arr[REASON][REPUTATION][idx] += 1
            elif tup.reason == 'course':
                arr[REASON][COURSE][idx] += 1
            else:  # 'other'
                arr[REASON][OTHER_REASON][idx] += 1
        elif c[0] == 'guardian':
            if tup.guardian == 'mother':
                arr[GUARDIAN][MOTHER][idx] += 1
            elif tup.guardian == 'father':
                arr[GUARDIAN][FATHER][idx] += 1
            else:  # 'other'
                arr[GUARDIAN][OTHER_GUARDIAN][idx] += 1
        elif c[0] == 'traveltime':
            if tup.traveltime == 'none':
                arr[TRAVELTIME][NONE][idx] += 1
            elif tup.traveltime == 'low':
                arr[TRAVELTIME][LOW][idx] += 1
            elif tup.traveltime == 'medium':
                arr[TRAVELTIME][MID][idx] += 1
            elif tup.traveltime == 'high':
                arr[TRAVELTIME][HIGH][idx] += 1
            else:  # 'very_high'
                arr[TRAVELTIME][VERY_HIGH][idx] += 1
        elif c[0] == 'studytime':
            if tup.studytime == 'none':
                arr[STUDYTIME][NONE][idx] += 1
            elif tup.studytime == 'low':
                arr[STUDYTIME][LOW][idx] += 1
            elif tup.studytime == 'medium':
                arr[STUDYTIME][MID][idx] += 1
            elif tup.studytime == 'high':
                arr[STUDYTIME][HIGH][idx] += 1
            else:  # 'very_high'
                arr[STUDYTIME][VERY_HIGH][idx] += 1
        elif c[0] == 'failures':
            if tup.failures == 'none':
                arr[FAILURES][NONE][idx] += 1
            elif tup.failures == 'low':
                arr[FAILURES][LOW][idx] += 1
            elif tup.failures == 'medium':
                arr[FAILURES][MID][idx] += 1
            elif tup.failures == 'high':
                arr[FAILURES][HIGH][idx] += 1
            else:  # 'very_high'
                arr[FAILURES][VERY_HIGH][idx] += 1
        elif c[0] == 'schoolsup':
            if tup.schoolsup == 'yes':
                arr[SCHOOLSUP][YES][idx] += 1
            else:  # 'no'
                arr[SCHOOLSUP][NO][idx] += 1
        elif c[0] == 'famsup':
            if tup.famsup == 'yes':
                arr[FAMSUP][YES][idx] += 1
            else:  # 'no'
                arr[FAMSUP][NO][idx] += 1
        elif c[0] == 'paid':
            if tup.paid == 'yes':
                arr[PAID][YES][idx] += 1
            else:  # 'no'
                arr[PAID][NO][idx] += 1
        elif c[0] == 'activities':
            if tup.activities == 'yes':
                arr[ACTIVITIES][YES][idx] += 1
            else:  # 'no'
                arr[ACTIVITIES][NO][idx] += 1
        elif c[0] == 'nursery':
            if tup.nursery == 'yes':
                arr[NURSERY][YES][idx] += 1
            else:  # 'no'
                arr[NURSERY][NO][idx] += 1
        elif c[0] == 'higher':
            if tup.higher == 'yes':
                arr[HIGHER][YES][idx] += 1
            else:  # 'no'
                arr[HIGHER][NO][idx] += 1
        elif c[0] == 'internet':
            if tup.internet == 'yes':
                arr[INTERNET][YES][idx] += 1
            else:  # 'no'
                arr[INTERNET][NO][idx] += 1
        elif c[0] == 'romantic':
            if tup.romantic == 'yes':
                arr[ROMANTIC][YES][idx] += 1
            else:  # 'no'
                arr[ROMANTIC][NO][idx] += 1
        elif c[0] == 'famrel':
            if tup.famrel == 1:
                arr[FAMREL][NONE][idx] += 1
            elif tup.famrel == 2:
                arr[FAMREL][LOW][idx] += 1
            elif tup.famrel == 3:
                arr[FAMREL][MID][idx] += 1
            elif tup.famrel == 4:
                arr[FAMREL][HIGH][idx] += 1
            else:  # 'excellent'
                arr[FAMREL][VERY_HIGH][idx] += 1
        elif c[0] == 'freetime':
            if tup.freetime == 1:
                arr[FREETIME][NONE][idx] += 1
            elif tup.freetime == 2:
                arr[FREETIME][LOW][idx] += 1
            elif tup.freetime == 3:
                arr[FREETIME][MID][idx] += 1
            elif tup.freetime == 4:
                arr[FREETIME][HIGH][idx] += 1
            else:  # 'excellent'
                arr[FREETIME][VERY_HIGH][idx] += 1
        elif c[0] == 'goout':
            if tup.goout == 1:
                arr[GOOUT][NONE][idx] += 1
            elif tup.goout == 2:
                arr[GOOUT][LOW][idx] += 1
            elif tup.goout == 3:
                arr[GOOUT][MID][idx] += 1
            elif tup.goout == 4:
                arr[GOOUT][HIGH][idx] += 1
            else:  # 'excellent'
                arr[GOOUT][VERY_HIGH][idx] += 1
        elif c[0] == 'Dalc':
            if tup.Dalc == 1:
                arr[DALC][NONE][idx] += 1
            elif tup.Dalc == 2:
                arr[DALC][LOW][idx] += 1
            elif tup.Dalc == 3:
                arr[DALC][MID][idx] += 1
            elif tup.Dalc == 4:
                arr[DALC][HIGH][idx] += 1
            else:  # 'excellent'
                arr[DALC][VERY_HIGH][idx] += 1
        elif c[0] == 'Walc':
            if tup.Walc == 1:
                arr[WALC][NONE][idx] += 1
            elif tup.Walc == 2:
                arr[WALC][LOW][idx] += 1
            elif tup.Walc == 3:
                arr[WALC][MID][idx] += 1
            elif tup.Walc == 4:
                arr[WALC][HIGH][idx] += 1
            else:  # 'excellent'
                arr[WALC][VERY_HIGH][idx] += 1
        elif c[0] == 'health':
            if tup.health == 1:
                arr[HEALTH][NONE][idx] += 1
            elif tup.health == 2:
                arr[HEALTH][LOW][idx] += 1
            elif tup.health == 3:
                arr[HEALTH][MID][idx] += 1
            elif tup.health == 4:
                arr[HEALTH][HIGH][idx] += 1
            else:  # 'excellent'
                arr[HEALTH][VERY_HIGH][idx] += 1
        elif c[0] == 'absences':
            if tup.absences == 'none':
                arr[ABSENCES][NONE][idx] += 1
            elif tup.absences == 'one_to_three':
                arr[ABSENCES][LOW][idx] += 1
            elif tup.absences == 'four_to_six':
                arr[ABSENCES][MID][idx] += 1
            elif tup.absences == 'seven_to_ten':
                arr[ABSENCES][HIGH][idx] += 1
            else:  # 'more_than_ten'
                arr[ABSENCES][VERY_HIGH][idx] += 1
        else:  # 'Grade'
            break

    return arr


# This function should build a supervised NB model. Returns the probability of each class and smoothed categorical distribution
def train(train_arg):
    row, column = train_arg.shape

    countY = np.zeros(CLASS)

    # Building an array to store the categorical distribution for all attribute values
    # I opted to use arrays because it's more intuitive to me compared to dictionaries
    catDist = np.zeros((column - 1, MAX_VALUE, CLASS))  # Unused elements are left zeros
    for r in train_arg.itertuples():
        if r.Grade == 'A+':
            countY[AA] += 1
            build_distribution(train_arg, r, catDist, AA)
        elif r.Grade == 'A':
            countY[A] += 1
            build_distribution(train_arg, r, catDist, A)
        elif r.Grade == 'B':
            countY[B] += 1
            build_distribution(train_arg, r, catDist, B)
        elif r.Grade == 'C':
            countY[C] += 1
            build_distribution(train_arg, r, catDist, C)
        elif r.Grade == 'D':
            countY[D] += 1
            build_distribution(train_arg, r, catDist, D)
        elif r.Grade == 'F':
            countY[F] += 1
            build_distribution(train_arg, r, catDist, F)
        else:
            print("Invalid Grade")
            exit()
    probY = np.zeros(CLASS)
    for i in range(CLASS):
        probY[i] = countY[i] / row

    possibleValue = {
        M2: [SCHOOL, SEX, ADDRESS, FAMSIZE, PSTATUS, SCHOOLSUP, FAMSUP, PAID, ACTIVITIES, NURSERY, HIGHER, INTERNET,
             ROMANTIC],
        M3: [GUARDIAN],
        M4: [MEDU, FEDU, REASON],
        M5: [MJOB, FJOB, TRAVELTIME, STUDYTIME, FAILURES, FAMREL, FREETIME, GOOUT, DALC, WALC, HEALTH, ABSENCES]}

    # Smoothing of categorical distribution array using Laplace smoothing
    catDistSmooth = np.zeros((column - 1, MAX_VALUE, CLASS))
    for x in range(column - 1):
        m = [k for k, v in possibleValue.items() if x in possibleValue[k]][0]
        for y in range(MAX_VALUE):
            if (y < m):
                for z in range(CLASS):
                    catDistSmooth[x, y, z] = ((ALPHA + catDist[x, y, z]) / (m * ALPHA + countY[z]))
    return probY, catDistSmooth


####################################################################
def build_prediction_arr(predArr_arg, psi_arg, col_idx, feature_idx, row_idx):
    for i in range(CLASS):
        predArr_arg[row_idx, i] *= psi_arg[col_idx, feature_idx, i]

    return


# This function should predict the class for an instance or a set of instances, based on a trained model
def predict(test_arg, classDist_arg, featureDist_arg):
    predictionArr = np.ones((test_arg.shape[0],
                             CLASS))  # 2D array to store the probabilities of students having each of the different grades
    indexArr = np.zeros(test_arg.shape[0]).astype(int)
    idx = 0

    for row in test_arg.itertuples():
        indexArr[idx] = row.Index

        if row.school == 'GP':
            build_prediction_arr(predictionArr, featureDist_arg, SCHOOL, GP, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, SCHOOL, MS, idx)

        if row.sex == 'F':
            build_prediction_arr(predictionArr, featureDist_arg, SEX, F_SEX, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, SEX, M_SEX, idx)

        if row.address == 'U':
            build_prediction_arr(predictionArr, featureDist_arg, ADDRESS, U, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, ADDRESS, R, idx)

        if row.famsize == "LE3":
            build_prediction_arr(predictionArr, featureDist_arg, FAMSIZE, LE3, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, FAMSIZE, GT3, idx)

        if row.Pstatus == 'T':
            build_prediction_arr(predictionArr, featureDist_arg, PSTATUS, T_PSTATUS, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, PSTATUS, A_PSTATUS, idx)

        if row.Medu == 'none':
            build_prediction_arr(predictionArr, featureDist_arg, MEDU, NONE, idx)
        elif row.Medu == 'low':
            build_prediction_arr(predictionArr, featureDist_arg, MEDU, LOW, idx)
        elif row.Medu == 'mid':
            build_prediction_arr(predictionArr, featureDist_arg, MEDU, MID, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, MEDU, HIGH, idx)

        if row.Fedu == 'none':
            build_prediction_arr(predictionArr, featureDist_arg, FEDU, NONE, idx)
        elif row.Fedu == 'low':
            build_prediction_arr(predictionArr, featureDist_arg, FEDU, LOW, idx)
        elif row.Fedu == 'mid':
            build_prediction_arr(predictionArr, featureDist_arg, FEDU, MID, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, FEDU, HIGH, idx)

        if row.Mjob == "teacher":
            build_prediction_arr(predictionArr, featureDist_arg, MJOB, TEACHER, idx)
        elif row.Mjob == "health":
            build_prediction_arr(predictionArr, featureDist_arg, MJOB, HEALTH_JOB, idx)
        elif row.Mjob == "services":
            build_prediction_arr(predictionArr, featureDist_arg, MJOB, SERVICES, idx)
        elif row.Mjob == "at_home":
            build_prediction_arr(predictionArr, featureDist_arg, MJOB, AT_HOME, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, MJOB, OTHER_JOB, idx)

        if row.Fjob == "teacher":
            build_prediction_arr(predictionArr, featureDist_arg, FJOB, TEACHER, idx)
        elif row.Fjob == "health":
            build_prediction_arr(predictionArr, featureDist_arg, FJOB, HEALTH_JOB, idx)
        elif row.Fjob == "services":
            build_prediction_arr(predictionArr, featureDist_arg, FJOB, SERVICES, idx)
        elif row.Fjob == "at_home":
            build_prediction_arr(predictionArr, featureDist_arg, FJOB, AT_HOME, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, FJOB, OTHER_JOB, idx)

        if row.reason == 'home':
            build_prediction_arr(predictionArr, featureDist_arg, REASON, HOME, idx)
        elif row.reason == 'reputation':
            build_prediction_arr(predictionArr, featureDist_arg, REASON, REPUTATION, idx)
        elif row.reason == 'course':
            build_prediction_arr(predictionArr, featureDist_arg, REASON, COURSE, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, REASON, OTHER_REASON, idx)

        if row.guardian == 'mother':
            build_prediction_arr(predictionArr, featureDist_arg, GUARDIAN, MOTHER, idx)
        elif row.guardian == 'father':
            build_prediction_arr(predictionArr, featureDist_arg, GUARDIAN, FATHER, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, GUARDIAN, OTHER_GUARDIAN, idx)

        if row.traveltime == 'none':
            build_prediction_arr(predictionArr, featureDist_arg, TRAVELTIME, NONE, idx)
        elif row.traveltime == 'low':
            build_prediction_arr(predictionArr, featureDist_arg, TRAVELTIME, LOW, idx)
        elif row.traveltime == 'mid':
            build_prediction_arr(predictionArr, featureDist_arg, TRAVELTIME, MID, idx)
        elif row.traveltime == 'high':
            build_prediction_arr(predictionArr, featureDist_arg, TRAVELTIME, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, TRAVELTIME, VERY_HIGH, idx)

        if row.studytime == 'none':
            build_prediction_arr(predictionArr, featureDist_arg, STUDYTIME, NONE, idx)
        elif row.studytime == 'low':
            build_prediction_arr(predictionArr, featureDist_arg, STUDYTIME, LOW, idx)
        elif row.studytime == 'mid':
            build_prediction_arr(predictionArr, featureDist_arg, STUDYTIME, MID, idx)
        elif row.studytime == 'high':
            build_prediction_arr(predictionArr, featureDist_arg, STUDYTIME, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, STUDYTIME, VERY_HIGH, idx)

        if row.failures == 'none':
            build_prediction_arr(predictionArr, featureDist_arg, FAILURES, NONE, idx)
        elif row.failures == 'low':
            build_prediction_arr(predictionArr, featureDist_arg, FAILURES, LOW, idx)
        elif row.failures == 'mid':
            build_prediction_arr(predictionArr, featureDist_arg, FAILURES, MID, idx)
        elif row.failures == 'high':
            build_prediction_arr(predictionArr, featureDist_arg, FAILURES, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, FAILURES, VERY_HIGH, idx)

        if row.schoolsup == 'yes':
            build_prediction_arr(predictionArr, featureDist_arg, SCHOOLSUP, YES, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, SCHOOLSUP, NO, idx)

        if row.famsup == 'yes':
            build_prediction_arr(predictionArr, featureDist_arg, FAMSUP, YES, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, FAMSUP, NO, idx)

        if row.paid == 'yes':
            build_prediction_arr(predictionArr, featureDist_arg, PAID, YES, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, PAID, NO, idx)

        if row.activities == 'yes':
            build_prediction_arr(predictionArr, featureDist_arg, ACTIVITIES, YES, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, ACTIVITIES, NO, idx)

        if row.nursery == 'yes':
            build_prediction_arr(predictionArr, featureDist_arg, NURSERY, YES, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, NURSERY, NO, idx)

        if row.higher == 'yes':
            build_prediction_arr(predictionArr, featureDist_arg, HIGHER, YES, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, HIGHER, NO, idx)

        if row.internet == 'yes':
            build_prediction_arr(predictionArr, featureDist_arg, INTERNET, YES, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, INTERNET, NO, idx)

        if row.romantic == 'yes':
            build_prediction_arr(predictionArr, featureDist_arg, ROMANTIC, YES, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, ROMANTIC, NO, idx)

        if row.famrel == 1:
            build_prediction_arr(predictionArr, featureDist_arg, FAMREL, NONE, idx)
        elif row.famrel == 2:
            build_prediction_arr(predictionArr, featureDist_arg, FAMREL, LOW, idx)
        elif row.famrel == 3:
            build_prediction_arr(predictionArr, featureDist_arg, FAMREL, MID, idx)
        elif row.famrel == 4:
            build_prediction_arr(predictionArr, featureDist_arg, FAMREL, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, FAMREL, VERY_HIGH, idx)

        if row.freetime == 1:
            build_prediction_arr(predictionArr, featureDist_arg, FREETIME, NONE, idx)
        elif row.freetime == 2:
            build_prediction_arr(predictionArr, featureDist_arg, FREETIME, LOW, idx)
        elif row.freetime == 3:
            build_prediction_arr(predictionArr, featureDist_arg, FREETIME, MID, idx)
        elif row.freetime == 4:
            build_prediction_arr(predictionArr, featureDist_arg, FREETIME, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, FREETIME, VERY_HIGH, idx)

        if row.goout == 1:
            build_prediction_arr(predictionArr, featureDist_arg, GOOUT, NONE, idx)
        elif row.goout == 2:
            build_prediction_arr(predictionArr, featureDist_arg, GOOUT, LOW, idx)
        elif row.goout == 3:
            build_prediction_arr(predictionArr, featureDist_arg, GOOUT, MID, idx)
        elif row.goout == 4:
            build_prediction_arr(predictionArr, featureDist_arg, GOOUT, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, GOOUT, VERY_HIGH, idx)

        if row.Dalc == 1:
            build_prediction_arr(predictionArr, featureDist_arg, DALC, NONE, idx)
        elif row.Dalc == 2:
            build_prediction_arr(predictionArr, featureDist_arg, DALC, LOW, idx)
        elif row.Dalc == 3:
            build_prediction_arr(predictionArr, featureDist_arg, DALC, MID, idx)
        elif row.Dalc == 4:
            build_prediction_arr(predictionArr, featureDist_arg, DALC, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, DALC, VERY_HIGH, idx)

        if row.Walc == 1:
            build_prediction_arr(predictionArr, featureDist_arg, WALC, NONE, idx)
        elif row.Walc == 2:
            build_prediction_arr(predictionArr, featureDist_arg, WALC, LOW, idx)
        elif row.Walc == 3:
            build_prediction_arr(predictionArr, featureDist_arg, WALC, MID, idx)
        elif row.Walc == 4:
            build_prediction_arr(predictionArr, featureDist_arg, WALC, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, WALC, VERY_HIGH, idx)

        if row.health == 1:
            build_prediction_arr(predictionArr, featureDist_arg, HEALTH, NONE, idx)
        elif row.health == 2:
            build_prediction_arr(predictionArr, featureDist_arg, HEALTH, LOW, idx)
        elif row.health == 3:
            build_prediction_arr(predictionArr, featureDist_arg, HEALTH, MID, idx)
        elif row.health == 4:
            build_prediction_arr(predictionArr, featureDist_arg, HEALTH, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, HEALTH, VERY_HIGH, idx)

        if row.absences == 'none':
            build_prediction_arr(predictionArr, featureDist_arg, ABSENCES, NONE, idx)
        elif row.absences == 'one_to_three':
            build_prediction_arr(predictionArr, featureDist_arg, ABSENCES, LOW, idx)
        elif row.absences == 'four_to_six':
            build_prediction_arr(predictionArr, featureDist_arg, ABSENCES, MID, idx)
        elif row.absences == 'seven_to_ten':
            build_prediction_arr(predictionArr, featureDist_arg, ABSENCES, HIGH, idx)
        else:
            build_prediction_arr(predictionArr, featureDist_arg, ABSENCES, VERY_HIGH, idx)

        idx += 1

    resultArr = np.zeros((test_arg.shape[0], 2)).astype(
        object)  # 2D array to store the maximum probability, hence the most probable grade
    for i in range(test_arg.shape[0]):
        max_prob = 0;
        max_index = 0;
        for j in range(CLASS):
            predictionArr[i][j] *= classDist_arg[j]
            if predictionArr[i][j] > max_prob:
                max_prob = predictionArr[i][j]
                max_index = j

        resultArr[i][0] = max_prob
        if max_index == 0:
            resultArr[i][1] = 'AA'
        elif max_index == 1:
            resultArr[i][1] = 'A'
        elif max_index == 2:
            resultArr[i][1] = 'B'
        elif max_index == 3:
            resultArr[i][1] = 'C'
        elif max_index == 4:
            resultArr[i][1] = 'D'
        else:
            resultArr[i][1] = 'F'

    resultList = {"Likelihood": resultArr[:, 0], "PredictedGrade": resultArr[:, 1]}
    resultDf = pd.DataFrame(resultList, index=indexArr[:])

    return resultDf


####################################################################
def evaluate(result_arg, testdata_arg):
    data_idx = 0
    correctlyLabelled = 0

    for student in result_arg.itertuples():
        if student.PredictedGrade == testdata_arg.iloc[data_idx, GRADE]:
            correctlyLabelled += 1
        accuracy = (correctlyLabelled / testdata_arg.shape[0])
        data_idx += 1

    return accuracy


####################################################################
PREDICTED_GRADE = 1


def compute_recall(method, prediction_arg, testset_arg):
    # In hindsight, probably shouldn't have used all capitals in order not to confuse these as constants
    TP_AA = 0
    TP_A = 0
    TP_B = 0
    TP_C = 0
    TP_D = 0
    TP_F = 0
    FN_AA = 0
    FN_A = 0
    FN_B = 0
    FN_C = 0
    FN_D = 0
    FN_F = 0

    data_idx = 0
    for instance in testset_arg.itertuples():
        if instance.Grade == "A+":
            if prediction_arg.iloc[data_idx, PREDICTED_GRADE] == "A+":
                TP_AA += 1
            else:
                FN_AA += 1
        elif instance.Grade == "A":
            if prediction_arg.iloc[data_idx, PREDICTED_GRADE] == "A":
                TP_A += 1
            else:
                FN_A += 1
        elif instance.Grade == "B":
            if prediction_arg.iloc[data_idx, PREDICTED_GRADE] == "B":
                TP_B += 1
            else:
                FN_B += 1
        elif instance.Grade == "C":
            if prediction_arg.iloc[data_idx, PREDICTED_GRADE] == "C":
                TP_C += 1
            else:
                FN_C += 1
        elif instance.Grade == "D":
            if prediction_arg.iloc[data_idx, PREDICTED_GRADE] == "D":
                TP_D += 1
            else:
                FN_D += 1
        elif instance.Grade == "F":
            if prediction_arg.iloc[data_idx, PREDICTED_GRADE] == "F":
                TP_F += 1
            else:
                FN_F += 1
        data_idx += 1

    if TP_AA + FN_AA == 0:
        recall_AA = 0
    else:
        recall_AA = TP_AA / (TP_AA + FN_AA)

    if TP_A + FN_A == 0:
        recall_A = 0
    else:
        recall_A = TP_A / (TP_A + FN_A)

    if TP_B + FN_B == 0:
        recall_B = 0
    else:
        recall_B = TP_B / (TP_B + FN_B)

    if TP_C + FN_C == 0:
        recall_C = 0
    else:
        recall_C = TP_C / (TP_C + FN_C)

    if TP_D + FN_D == 0:
        recall_D = 0
    else:
        recall_D = TP_D / (TP_D + FN_D)

    if TP_F + FN_F == 0:
        recall_F = 0
    else:
        recall_F = TP_F / (TP_F + FN_F)

    if method == "macroaverage":
        recall = ((recall_AA + recall_A + recall_B + recall_C + recall_D + recall_F) / 6)

    elif method == "separately":
        recallArr = [0 for x in range(6)]
        recallArr[AA] = recall_AA
        recallArr[A] = recall_A
        recallArr[B] = recall_B
        recallArr[C] = recall_C
        recallArr[D] = recall_D
        recallArr[F] = recall_F
        return recallArr

    else:
        print("Please enter 'separately' or 'macroaverage'")

    return recall


def compute_precision(method, prediction_arg, testset_arg):
    TP_AA = 0
    TP_A = 0
    TP_B = 0
    TP_C = 0
    TP_D = 0
    TP_F = 0
    FP_AA = 0
    FP_A = 0
    FP_B = 0
    FP_C = 0
    FP_D = 0
    FP_F = 0

    data_idx = 0
    for instance in prediction_arg.itertuples():
        if instance.PredictedGrade == "A+":
            if testset_arg.iloc[data_idx, GRADE] == "A+":
                TP_AA += 1
            else:
                FP_AA += 1
        elif instance.PredictedGrade == "A":
            if testset_arg.iloc[data_idx, GRADE] == "A":
                TP_A += 1
            else:
                FP_A += 1
        elif instance.PredictedGrade == "B":
            if testset_arg.iloc[data_idx, GRADE] == "B":
                TP_B += 1
            else:
                FP_B += 1
        elif instance.PredictedGrade == "C":
            if testset_arg.iloc[data_idx, GRADE] == "C":
                TP_C += 1
            else:
                FP_C += 1
        elif instance.PredictedGrade == "D":
            if testset_arg.iloc[data_idx, GRADE] == "D":
                TP_D += 1
            else:
                FP_D += 1
        elif instance.PredictedGrade == "F":
            if testset_arg.iloc[data_idx, GRADE] == "F":
                TP_F += 1
            else:
                FP_F += 1
        data_idx += 1

    if TP_AA + FP_AA == 0:
        precision_AA = 0
    else:
        precision_AA = TP_AA / (TP_AA + FP_AA)

    if TP_A + FP_A == 0:
        precision_A = 0
    else:
        precision_A = TP_A / (TP_A + FP_A)

    if TP_B + FP_B == 0:
        precision_B = 0
    else:
        precision_B = TP_B / (TP_B + FP_B)

    if TP_C + FP_C == 0:
        precision_C = 0
    else:
        precision_C = TP_C / (TP_C + FP_C)

    if TP_D + FP_D == 0:
        precision_D = 0
    else:
        precision_D = TP_D / (TP_D + FP_D)

    if TP_F + FP_F == 0:
        precision_F = 0
    else:
        precision_F = TP_F / (TP_F + FP_F)

    if method == "macroaverage":
        precision = ((precision_AA + precision_A + precision_B + precision_C + precision_D + precision_F) / 6)

    elif method == "separately":
        precisionArr = [0 for x in range(6)]
        precisionArr[AA] = precision_AA
        precisionArr[A] = precision_A
        precisionArr[B] = precision_B
        precisionArr[C] = precision_C
        precisionArr[D] = precision_D
        precisionArr[F] = precision_F
        return precisionArr

    else:
        print("Please enter 'separately' or 'macroaverage'")
        exit(0)

    return precision


def compute_f1_macroaverage(recall, precision):
    if precision + recall != 0:
        f1 = (2 * recall * precision) / (precision + recall)
    else:
        f1 = 0
    return f1


def compute_f1_separately(recallArr, precisionArr):
    f1Arr = [0 for x in range(6)]

    for i in range(6):
        if precisionArr[i] + recallArr[i] != 0:
            f1Arr[i] = (2 * recallArr[i] * precisionArr[i]) / (precisionArr[i] + recallArr[i])
        else:
            f1Arr[i] = 0
    return f1Arr


####################################################################
def cross_validation(data_set):
    prediction_Arr = np.zeros((data_set.shape[0], 2)).astype(object)
    predictionDf = pd.DataFrame(columns=["Likelihood", "PredictedGrade"])
    for instance in data_set.itertuples():  # A leave-one-out strategy was chosen to maximise training data
        instance_frame = {'school': instance.school, 'sex': instance.sex, 'address': instance.address,
                          'famsize': instance.famsize, 'Pstatus': instance.Pstatus, 'Medu': instance.Medu,
                          'Fedu': instance.Fedu, 'Mjob': instance.Mjob, 'Fjob': instance.Fjob,
                          'reason': instance.reason, 'guardian': instance.guardian, 'traveltime': instance.traveltime,
                          'studytime': instance.studytime, 'failures': instance.failures,
                          'schoolsup': instance.schoolsup, 'famsup': instance.famsup, 'paid': instance.paid,
                          'activities': instance.activities, 'nursery': instance.nursery, 'higher': instance.higher,
                          'internet': instance.internet, 'romantic': instance.romantic, 'famrel': instance.famrel,
                          'freetime': instance.freetime, 'goout': instance.goout, 'Dalc': instance.Dalc,
                          'Walc': instance.Walc, 'health': instance.health, 'absences': instance.absences,
                          'Grade': instance.Grade}
        instanceDf = pd.DataFrame(instance_frame, index=[instance.Index])
        data_cpy = data_set.copy(deep=True)
        data_cpy.drop([instance.Index], inplace=True)
        phi, psi = train(data_cpy)
        predictionResult = predict(instanceDf, phi, psi)
        predictionDf = predictionDf.append(predictionResult)
    accuracy = evaluate(predictionDf, data_set)

    return accuracy