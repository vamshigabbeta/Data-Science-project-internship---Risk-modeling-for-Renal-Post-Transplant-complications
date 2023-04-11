# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 02:16:38 2023

@author: VAMSHI
"""

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

loaded_model = pickle.load(open(r'C:/Users/DELL/Downloads/rsf_model_final','rb'))
 


# Specify the title and logo for the web page.
st.set_page_config(page_title='Graft Survival Predictor',
                   page_icon = 'https://st4.depositphotos.com/4177785/23186/v/450/depositphotos_231861846-stock-illustration-smiling-human-kidneys-color-vector.jpg',
                   layout="wide")


#Specify the title
st.title('GRAFT SURVIVAL PREDICTOR')


# Add a sidebar to the web page. 
st.markdown('---')
# Sidebar Configuration
st.sidebar.image('https://lucknow.apollohospitals.com/wp-content/uploads/2021/09/kidney-transplant2.jpg', width=200)
st.sidebar.markdown('GRAFT SURVIIVAL PREDICTION')
st.sidebar.markdown('minimizing the renal post transplant complications')
st.sidebar.markdown('We can predict the graft survival for the candidate using this app') 

st.sidebar.markdown('---')
st.sidebar.write('Developed by Gabbeta vamshi')

        

#Creating a function for prediction

def survival_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    
    surv_test_rsf = loaded_model.predict_survival_function(input_data_reshaped, return_array=False)
    surv = loaded_model.predict_survival_function(input_data_reshaped, return_array=True)

    event_times = loaded_model.event_times_

    lower, upper = event_times[0], event_times[-1]
    y_times = np.arange(lower, upper)

    T1, T2 = surv_test_rsf[0].x.min(),surv_test_rsf[0].x.max()
    mask = np.logical_or(y_times >= T2, y_times < T1) # mask outer interval
    times = y_times[~mask]

    rsf_surv_prob = np.row_stack([ fn(times) for fn in surv_test_rsf])
    rsf_surv_prob_test1 = pd.DataFrame(rsf_surv_prob)
    
    six_months = rsf_surv_prob_test1.iloc[ :,24]*100
    one_year = rsf_surv_prob_test1.iloc[ :,52]*100
    two_years = rsf_surv_prob_test1.iloc[ :,104]*100 
    five_years = rsf_surv_prob_test1.iloc[ :,260]*100 
    
    print('Survival Probability at 6 months in %:',(rsf_surv_prob_test1.iloc[ :,24])*100)
    
    print('Survival Probability at 1 year in %:',(rsf_surv_prob_test1.iloc[ :,52])*100)
    
    print('Survival Probability at 2 years in %:',(rsf_surv_prob_test1.iloc[ :,104])*100)
    
    print('Survival Probability at 5 years in %:',(rsf_surv_prob_test1.iloc[ :,260])*100)

    plt.step(loaded_model.event_times_,surv[0], where="post", label=str(0))
    plt.ylabel("Survival probability")
    plt.xlabel("Time in weeks")
    plt.legend()
    plt.grid(True)
    st.pyplot()
    
    return(('Probability of survival at 6 months in % :', six_months.to_string(index=False)),
    ('Probability of survival at 1 year in % :', one_year.to_string(index=False)),
    ('Probability of survival at 2 years in % :', two_years.to_string(index=False)), 
    ('Probability of survival at 5 years in % :', five_years.to_string(index=False))) 



def main():
     
     
     #Giving a title
      
     st.title('Graft survival Prediction web app') #giving title
     
     
     #Getting input data from user
     Systolicbloodpressure = st.number_input('Systolic Blood Pressure of the patient' )
     Diastolicbloodpressure = st.number_input('Diastolic Blood Pressure of the patient:' )
     Whitebloodcellcount = st.number_input('White blood cell count:')
     Hemoglobinlevel	= st.number_input('Hemoglobin level:')
     Platelets = st.number_input('Platelets count:' )
     Serumcreatininelevel = st.number_input('Serum Creatinine level:')
     BloodUreaNitrogenlevel = st.number_input('Blood Urea Nitrogen level:')
     Glucoselevel = st.number_input('Level of glucose:' )
     Potassiumlevel = st.number_input('Level of Potassium:' )
     Sodiumlevel = st.number_input('Sodium level:' )
     Calciumlevel = st.number_input('Calcium level:' )
     Phosphoruslevel	= st.number_input('Phosphorous level:' )
     Tac_MR = st.number_input('Tacrolimus level:')
     Recipientsage = st.number_input('Age of the recipients:' )
     Donorage = st.number_input('Age of the Donor:' )
     Bodymassindex = st.number_input('Body Mass Index of the patient:')
     Numberofposttransplantadmission	= st.number_input('Number of times admitted after transplantation:' )
     Durationsincetransplant	= st.number_input('Duration since transplant:' )
     eGFR = st.number_input('Glomerular Filteration Rate:')
     
     Recipientssex =  st.radio("Gender(Recipient):", ('Male', 'Female'))
     if Recipientssex == "Male":
         Recipientssex = 0
     else:
         Recipientssex = 1
         
     RecipientsReligion =st.radio("Religion(Recipient):",('Orthodox','Muslim', 'Protestant' ,'Catholic','Others')) 
     if RecipientsReligion == 'Orthodox':
         RecipientsReligion = 0
     elif RecipientsReligion == 'Muslim':
         RecipientsReligion = 1
     elif RecipientsReligion == 'Protestant':
         RecipientsReligion = 2
     elif RecipientsReligion == 'Catholic':
         RecipientsReligion = 3
     elif RecipientsReligion == 'Others':
         RecipientsReligion = 4
             
     Recipientslevelofeducation = st.radio('Education level(Recipient):', ('No education', 'Primary','Secondary','Tertiary(Diploma)','Degree and above'))
     if Recipientslevelofeducation == 'No education':
         Recipientslevelofeducation = 0
     elif Recipientslevelofeducation == 'Primary':
         Recipientslevelofeducation = 1
     elif Recipientslevelofeducation == 'Secondary':
         Recipientslevelofeducation = 2
     elif Recipientslevelofeducation == 'Tertiary(Diploma)':
         Recipientslevelofeducation = 3
     elif Recipientslevelofeducation == 'Degree and above':
         Recipientslevelofeducation = 4
         
     Recipientsemploymentstatus = st.radio('Employment Status(Recipient):', ('Government employee','Private employee','Not working'))	
     if Recipientsemploymentstatus == 'Government employee':
          Recipientsemploymentstatus = 0
     elif Recipientsemploymentstatus == 'Private employee':
          Recipientsemploymentstatus = 1
     elif Recipientsemploymentstatus == 'Not working':
          Recipientsemploymentstatus = 2
          
     Recipientsresidence = st.radio('Residence(Recipient):', ('Urban','Rural'))
     if Recipientsresidence == 'Urban':
         Recipientsresidence = 0
     else:
         Recipientsresidence = 1
         
     Donorsex = st.radio("Gender(Donor):", ('Male', 'Female'))
     if Donorsex == "Male":
            Donorsex = 0
     else:
            Donorsex = 1  
            
     Donortorecipientrelationship = st.radio('Relationship between Donor and Recipient:', ('Sibling','Parent','Child', 'Spouse', 'Relatives'))
     if Donortorecipientrelationship == 'Sibling':
            Donortorecipientrelationship = 0
     elif Donortorecipientrelationship == 'Parent':
            Donortorecipientrelationship = 1
     elif Donortorecipientrelationship == 'Child':
            Donortorecipientrelationship = 2
     elif Donortorecipientrelationship == 'Spouse':
             Donortorecipientrelationship = 3
     elif Donortorecipientrelationship == 'Relatives':
             Donortorecipientrelationship = 4
             
             
             
     Placecenterofallograft = st.radio('Allograft centre:', ('Locally in the center','Outside the country'))
     if Placecenterofallograft == 'Locally in the center':
         Placecenterofallograft = 0
     else:
         Placecenterofallograft = 1  
     
     Posttransplantregularphysicale = st.radio('Regular Physical Exercise after transplant:', ('Yes','No'))
     if Posttransplantregularphysicale == "No":
         Posttransplantregularphysicale = 0
     else:
         Posttransplantregularphysicale = 1  
         
     Pretransplanthistoryofsubstanc = st.radio('Pre-Transplant History of substance abuse', ('Yes','No'))
     if Pretransplanthistoryofsubstanc == "No":
         Pretransplanthistoryofsubstanc = 0
     else:
         Pretransplanthistoryofsubstanc = 1  
     
     Posttransplantnonadherence = st.radio('Non Adherence after transplant:', ('Yes','No'))
     if Posttransplantnonadherence == "No":
         Posttransplantnonadherence = 0
     else:
         Posttransplantnonadherence = 1  
         
     CausesofEndStageRenalDisea = st.radio('Causes of End Stage Renal Disease:', ('Chronic glomerulonephritis','Diabetes','Hypertension','Others','Not determined/Unknown'))
     if CausesofEndStageRenalDisea == 'Chronic glomerulonephritis':
            CausesofEndStageRenalDisea = 0
     elif CausesofEndStageRenalDisea == 'Diabetes':
            CausesofEndStageRenalDisea = 1
     elif CausesofEndStageRenalDisea == 'Hypertension':
            CausesofEndStageRenalDisea = 2
     elif CausesofEndStageRenalDisea == 'Others':
            CausesofEndStageRenalDisea = 3
     elif CausesofEndStageRenalDisea == 'Not determined/Unknown':
            CausesofEndStageRenalDisea = 4
            
     Historyofpretransplantcomorbid = st.radio('History of Comorbidities before transplant:', ('Yes','No'))
     if Historyofpretransplantcomorbid == "No":
         Historyofpretransplantcomorbid = 0
     else:
         Historyofpretransplantcomorbid = 1  
         
     Historyofdialysisbeforetranspl = st.radio('Dialysis History before transplant:', ('Yes','No'))
     if Historyofdialysisbeforetranspl == "No":
         Historyofdialysisbeforetranspl = 0
     else:
         Historyofdialysisbeforetranspl = 1  
         
     Historyofbloodtransfusi	= st.radio('Blood Transfusion History:', ('Yes','No'))
     if Historyofbloodtransfusi == "No":
         Historyofbloodtransfusi = 0
     else:
         Historyofbloodtransfusi = 1  
         
     Historyofabdominalsurge = st.radio('History of Abdominal Surge:', ('Yes','No'))
     if Historyofabdominalsurge == "No":
         Historyofabdominalsurge = 0
     else:
         Historyofabdominalsurge = 1  
         
     Familyhistoryofkidneydisea = st.radio('History of Kidney Disease in Family:', ('Yes','No'))
     if Familyhistoryofkidneydisea == "No":
         Familyhistoryofkidneydisea = 0
     else:
         Familyhistoryofkidneydisea = 1  
         
     Posttransplantmalignan = st.radio('Malignancy After Transplantation:', ('Yes','No'))
     if Posttransplantmalignan == "No":
         Posttransplantmalignan = 0
     else:
         Posttransplantmalignan = 1  
         
     PosttransplantUrologicalcompli = st.radio('Urological complication After Transplantation:', ('Yes','No')) 
     if PosttransplantUrologicalcompli == "No":
         PosttransplantUrologicalcompli = 0
     else:
         PosttransplantUrologicalcompli = 1  
         
     PosttransplantVascularcomplica = st.radio('Vascular complication After Transplantation:', ('Yes','No'))
     if PosttransplantVascularcomplica == "No":
         PosttransplantVascularcomplica = 0
     else:
         PosttransplantVascularcomplica = 1  
         
     PosttransplantCardiovascularco = st.radio('Cardiovascular complication After Transplantation:', ('Yes','No'))
     if PosttransplantCardiovascularco == "No":
         PosttransplantCardiovascularco = 0
     else:
         PosttransplantCardiovascularco = 1  
         
     PosttransplantInfection = st.radio('Infections After Transplantation:', ('Yes','No'))
     if PosttransplantInfection == "No":
         PosttransplantInfection = 0
     else:
         PosttransplantInfection = 1  
         
     Posttransplantdiabetes = st.radio('Diabetes after Transplantation:', ('Yes','No'))
     if Posttransplantdiabetes == "No":
         Posttransplantdiabetes = 0
     else:
         Posttransplantdiabetes = 1  
         
     Posttransplanthypertension = st.radio('Hypertension after Transplantation:', ('Yes','No')) 
     if Posttransplanthypertension == "No":
         Posttransplanthypertension = 0
     else:
         Posttransplanthypertension = 1  
         
     Anepisodeofacuterejection = st.radio('Acute Rejection:', ('Yes','No'))
     if Anepisodeofacuterejection == "No":
         Anepisodeofacuterejection = 0
     else:
         Anepisodeofacuterejection = 1  
         
     Anepisodeofchronicrejection = st.radio('Chronic Rejection:', ('Yes','No'))
     if Anepisodeofchronicrejection == "No":
         Anepisodeofchronicrejection = 0
     else:
         Anepisodeofchronicrejection = 1  	
     
     PosttransplantGastrointestin = st.radio('Gastro-intestinal complication:', ('Yes','No'))
     if PosttransplantGastrointestin == "No":
         PosttransplantGastrointestin = 0
     else:
         PosttransplantGastrointestin = 1  	
     
     Posttransplantglomerulonephrit = st.radio('Glomerulonephritis after transplant:', ('Yes','No'))
     if  Posttransplantglomerulonephrit == "No":
          Posttransplantglomerulonephrit = 0
     else:
          Posttransplantglomerulonephrit = 1  	
         
     Posttransplantdelayedgraftfunc = st.radio('Delay of graft function after transplant:', ('Yes','No')) 
     if Posttransplantdelayedgraftfunc == "No":
         Posttransplantdelayedgraftfunc = 0
     else:
         Posttransplantdelayedgraftfunc = 1  	
     
     Posttransplantfluidoverloa = st.radio('Overloading of fluids after transplant:', ('Yes','No'))
     if  Posttransplantfluidoverloa == "No":
          Posttransplantfluidoverloa = 0
     else:
          Posttransplantfluidoverloa = 1  
         
     PosttransplantCovid19 = st.radio('Covid19 after transplant:', ('Yes','No'))
     if PosttransplantCovid19 == "No":
         PosttransplantCovid19 = 0
     else:
         PosttransplantCovid19 = 1  	
     
     marital_statuss	= st.radio('Marital Status:', ('Married','Unmarried')) 
     if  marital_statuss == 'Married':
          marital_statuss = 0
     else:
          marital_statuss = 1  	
     
     postwaterintakee =  st.radio('Water intake post transplantation(in litres):',('Less than 2 litres','2-3 litres','3-4 litres','More than 4 litres'))
     if postwaterintakee == 'Less than 2 litres':
            postwaterintakee = 0
     elif postwaterintakee == '2-3 litres':
            postwaterintakee = 1
     elif postwaterintakee == '3-4 litres':
            postwaterintakee= 2
     elif postwaterintakee == 'More than 4 litres':
            postwaterintakee= 3
  
     #Code for prediction
     result = ''
     
     #Creating a button for prediction
     if st.button('Predict'):
         result = survival_prediction([Systolicbloodpressure, Diastolicbloodpressure, Whitebloodcellcount,	
                             Hemoglobinlevel, Platelets, Serumcreatininelevel, BloodUreaNitrogenlevel,	
                             Glucoselevel, Potassiumlevel, Sodiumlevel, Calciumlevel, Phosphoruslevel,	
                             Tac_MR, Recipientsage, Donorage, Bodymassindex,	Numberofposttransplantadmission,	
                             Durationsincetransplant, eGFR, Recipientssex, RecipientsReligion, 
                             Recipientslevelofeducation, Recipientsemploymentstatus,	Recipientsresidence,
                             Donorsex, Donortorecipientrelationship, Placecenterofallograft, 
                             Posttransplantregularphysicale, Pretransplanthistoryofsubstanc,	
                             Posttransplantnonadherence, CausesofEndStageRenalDisea,	
                             Historyofpretransplantcomorbid, Historyofdialysisbeforetranspl, Historyofbloodtransfusi,	
                             Historyofabdominalsurge, Familyhistoryofkidneydisea, Posttransplantmalignan,	
                             PosttransplantUrologicalcompli, PosttransplantVascularcomplica,
                             PosttransplantCardiovascularco, PosttransplantInfection, Posttransplantdiabetes,	
                             Posttransplanthypertension, Anepisodeofacuterejection, Anepisodeofchronicrejection,	
                             PosttransplantGastrointestin, Posttransplantglomerulonephrit,	
                             Posttransplantdelayedgraftfunc, Posttransplantfluidoverloa, PosttransplantCovid19, 
                             marital_statuss, postwaterintakee])
         
     st.success(result)    
     
     

if __name__ == '__main__':
     main()
     