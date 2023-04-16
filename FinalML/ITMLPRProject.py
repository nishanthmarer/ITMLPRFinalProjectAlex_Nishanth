"""///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Authors: Nishant Marer Prabhu, Alexander Montes McNiel
NUID: 002624650
FileName: ITMLPRProject.py
Semester: Spring 2023
Class: Indroduction to Machine Learning and Pattern Recognition
Description: This file is responsible for reading the data from a processed csv file placed in the same location as the script and then using the 
ERM technique the labels for each of the data point is predicted

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""
import pandas as pd
import numpy as np
import numpy.matlib
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

"""This function is responsible for scaling the data to the interval [0,1] using the MinMaxScalar"""
def scaleData(DataFrame_NumpyArray):
    scaler = MinMaxScaler()
    DataFrame = pd.DataFrame(DataFrame_NumpyArray)
    
    scaler.fit(DataFrame)
    scaled = scaler.fit_transform(DataFrame)
    scaled_df = pd.DataFrame(scaled, columns=DataFrame.columns)
    
    return scaled_df.to_numpy().T

"""This function reads the data from the csv file and drops the unnecessary columns and keep only the required columns"""
def readDatafromDisk(l_pLocation, l_pLabelColumn_Drop, label_Column_Keep, resultColumnName):
    
    datafromdisk = pd.read_csv(l_pLocation)
    datafromdisk = datafromdisk.drop(labels=l_pLabelColumn_Drop, axis='columns')
    
    datafromdisk = datafromdisk.sort_values(by=[resultColumnName])
    X_train = datafromdisk[label_Column_Keep]
    y_train = datafromdisk[resultColumnName]
    
    FeaturesColumnNames = X_train.columns.values.tolist()
    
    X_2DArray_Features = (X_train.T).to_numpy()
    Y_1DArray_Labels = y_train.to_numpy()
    
    return X_2DArray_Features,Y_1DArray_Labels, FeaturesColumnNames

"""This function finds the label count"""
def find_LabelCount(l_pLabelSet, l_pY_LabelSet):
    
    Label_Count = []
    for i in range(0,len(l_pLabelSet)):
        x = len([j for j in range(0,len(l_pY_LabelSet)) if (l_pY_LabelSet[j] == l_pLabelSet[i])])
        Label_Count.append(x)
        x = 0
    
    return Label_Count

"""This function finds the Mean and Cov matrix for each of the labels"""
def findMeanAndCovarience(l_pData, FeaturesColumnNames):
    
    """Convert back to dataframe as it is easy to compute mean and covarience"""
    
    DataFrame_ParticularLabel = pd.DataFrame(l_pData)
    Mean_Vector = (DataFrame_ParticularLabel.mean(axis=1)).to_numpy()
    Covarience_Vector = ((DataFrame_ParticularLabel.T).cov()).to_numpy()
    
    return Mean_Vector, Covarience_Vector

"""This function takes in the dataset and finds the Mean and Cov matrix"""
def determineMetricsforDataset(l_pData, Label_Count,FeaturesColumnNames):
    
    SumOfRows = 0
    Mean_Vector = []
    Covarience_Matrix = []
    LengthofFeatures = len(FeaturesColumnNames)
    
    for i in range(0,len(Label_Count)):
        
        XData = l_pData[:, SumOfRows: (SumOfRows + Label_Count[i])]
        SumOfRows = SumOfRows + Label_Count[i]
        if XData.size == 0:
            Mean_Vector.append(np.zeros(shape=LengthofFeatures))
            Covarience_Matrix.append(np.zeros((LengthofFeatures,LengthofFeatures)))
        else:     
            Mean_Vector_Label, Covarience_Mat_Label = findMeanAndCovarience(XData,FeaturesColumnNames)
            Mean_Vector.append(Mean_Vector_Label)
            Covarience_Matrix.append(Covarience_Mat_Label)
            
    return Mean_Vector, Covarience_Matrix

"""This function finds the class priors"""
def findClassPrior(l_pLabel_Count):
    
    class_prior = []
    SumOfLabelsCount = sum(l_pLabel_Count)
    for label in l_pLabel_Count:
        class_prior.append(label/SumOfLabelsCount)
    
    return class_prior

"""Evaluate the Gaussian for the given data samples and mean, covarience matrix"""
def evalulate_PDFGaussian(l_pDataSamples,l_pMeanVector, l_pCovMatrix):
    pdf_evaluation = (multivariate_normal.pdf(l_pDataSamples.T,mean=l_pMeanVector, cov = l_pCovMatrix))
    return pdf_evaluation

"""This function evaluates the Multiple PDF"""
def evaluate_MultiplePDF(X_FeatureSet, Label_Count, LengthofFeatures, Sample_Count, Mean_Vector, Covarience_Matrix):
    
    Evaluated_PDF_Data = np.zeros((LengthofFeatures,Sample_Count))
    for i in range(0,len(Label_Count)):
        if sum(Mean_Vector[i]) != 0:
            Evaluated_PDF_Data[i,:] = evalulate_PDFGaussian(X_FeatureSet, Mean_Vector[i], Covarience_Matrix[i])
        else:
            Evaluated_PDF_Data[i,:] = np.zeros(Sample_Count)
    return Evaluated_PDF_Data

"""This function is responsible for multiplying the loss matrix with the class posterior value"""   
def multiply_LossMatrices(LossMatrix, MatrixB, l_pSamplesCount, l_pNumOfDim):
    MatrixProd = np.zeros((l_pNumOfDim,l_pSamplesCount))
    
    for k in range(0,l_pNumOfDim):
        for j in range(0,len(MatrixB[0])): 
            for i in range(0,l_pNumOfDim):
                MatrixProd[k,j] += LossMatrix[k][i] * MatrixB[i][j]
    
    return MatrixProd

"""This function determines the lowest of the risk value for a particular decision"""
def findDecisionMade(RiskMatrix):    
    DecisionList = []
    for i in range(0,RiskMatrix.shape[1]):
        DecisionList.append(np.argmin(np.array(RiskMatrix[:,i])))
    
    return DecisionList

"""This function determines the confusion matrix"""
def getConfusionMatrix(DecisionList, l_pDecisionLabels, l_pClassLabels, l_pNumofDim, l_pClassLabel):
    
    Confusion_Matrix = np.zeros((l_pNumofDim,l_pNumofDim))
    for DC in l_pDecisionLabels:
        for CL in l_pClassLabel:
            
            Decision_Label_DL = [k for k in range(len(l_pClassLabels)) if (l_pClassLabels[k] == CL and DecisionList[k] == DC)]
            Label_Count = [m for m in range(len(l_pClassLabels)) if (l_pClassLabels[m] == CL)]
            if len(Label_Count) == 0:
                Confusion_Matrix[DC][CL] = 0
            else:
                Confusion_Matrix[DC][CL] = len(Decision_Label_DL)/len(Label_Count)
            Label_Count = 0
            Decision_Label_DL = 0
    
    return Confusion_Matrix

"""This function uses the confusion matrix to determine the PminError"""
def getMinPError(ConfusionMatrix):
    Sum = 0
    Sum_Diag = 0
    for i in range(0,len(ConfusionMatrix)):
        Sum_Diag = Sum_Diag + ConfusionMatrix[i][i]
        for j in range(0,len(ConfusionMatrix)):
           Sum = Sum +  ConfusionMatrix[i][j]
           
    
    Accuracy = 0
    Accuracy = float(Sum_Diag)/float(Sum)
    return ( 1 - Accuracy)

"""This function regularises the Covarience matrix so that it can be inverted"""
def regulariseCovarienceMatrix(Lambda,Covarience_Matrix, numofDims):
    
    cov_RegMatrix = []
    for matrix in Covarience_Matrix:
        cov_RegMatrix.append(matrix + Lambda*np.identity(numofDims))
        
    return cov_RegMatrix

"""Main function"""
def main():
    
    InputData = 'data_preprocessed.csv'
    label_Column_drop = ['schedule_date','schedule_season','schedule_week','team_home','team_away','score_home','score_away','team_favorite_id','over_under_line','stadium','stadium_neutral','weather_temperature','weather_wind_mph','weather_humidity','weather_detail','over','team_home_division','team_away_division','division_game']
    label_Column_Keep = ['schedule_playoff','spread_favorite','home_favorite','away_favorite','team_away_current_win_pct','team_home_current_win_pct','team_home_lastseason_win_pct','team_away_lastseason_win_pct','elo_prob1','elo_prob2']

    resultColumnName = 'result'
    LabelSet = [0,1]
    DecisionSet = [0,1]

    """Read the data set"""
    X_FeatureSet, Y_LabelSet, FeaturesColumnNames = readDatafromDisk(InputData,label_Column_drop,label_Column_Keep,resultColumnName)
    OutputDimensionLength = 2
    
    Featureset = len(label_Column_Keep)
    #X_FeatureSet = scaleData(X_FeatureSet.T)
    
    Label_Count = find_LabelCount(LabelSet, Y_LabelSet)
    Sample_Count = sum(Label_Count)
    
    Mean_Vector = []
    Covarience_Matrix = []
    Mean_Vector, Covarience_Matrix = determineMetricsforDataset(X_FeatureSet,Label_Count, FeaturesColumnNames)
    
    """Find the class Priors for each label"""
    Class_Prior = findClassPrior(Label_Count)
    
    """Scale the covarience matrix so that the input matrix is symmetric positive definite"""
    """CRegularized = CSampleAverage +λI where λ > 0"""
    Lambda_Scale = 0.000005
    Regularised_Cov_Matrix = regulariseCovarienceMatrix(Lambda_Scale, Covarience_Matrix, Featureset)
    
    """Evaluate the PDF"""
    Evaluated_PDF_Data = np.zeros((OutputDimensionLength,Sample_Count))
    Evaluated_PDF_Data = evaluate_MultiplePDF(X_FeatureSet, Label_Count, OutputDimensionLength, Sample_Count, Mean_Vector, Regularised_Cov_Matrix)
    
    """Calculate the P(x)"""
    Px = np.matmul(Class_Prior, Evaluated_PDF_Data)
    
    """Stack the class_Prior to a horizontal structure and create 10k columns of it for multiplication with the class pdf"""
    Class_Prior_VStack = numpy.vstack(Class_Prior)
    class_Priors_VStack_Multiple = np.matlib.repmat(Class_Prior_VStack, 1, Sample_Count)
    
    """Determine the class posterior"""
    ClassPosterior =  np.divide((np.multiply(Evaluated_PDF_Data,class_Priors_VStack_Multiple)),Px)
    
    """Create a loss matrix for the MAP Classifier"""
    loss_matrix = np.ones((OutputDimensionLength, OutputDimensionLength)) - np.identity(OutputDimensionLength)
    
    """Multiplty the Loss Matrix with the Class Posterior"""
    RiskMatrix = multiply_LossMatrices(loss_matrix,ClassPosterior,Sample_Count,OutputDimensionLength)
    
    """Determine the list risk value for each of the decision made column wise (As the data is stacked as a column vector)"""
    DecisionList = []
    DecisionList = findDecisionMade(RiskMatrix)
    
    """Get the confusion Matrix"""
    Confusion_Matrix = getConfusionMatrix(DecisionList, DecisionSet, Y_LabelSet, OutputDimensionLength, LabelSet)
    print('The confusion matrix for the 0-1 loss Map Classifier is as follows')
    print(Confusion_Matrix)
    print('\n')
    
    PminError = getMinPError(Confusion_Matrix)
    print("The PminError for the Lambda01 loss matrix is: ", PminError)

    """Heat Map of the confusion matrix"""
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(Confusion_Matrix, annot=True, fmt='.2f')
    plt.ylabel('Decisions')
    plt.xlabel('Labels',loc='center')
    plt.show(block=False)
  
"""Call the main function"""      
if __name__ == "__main__":
    main()          