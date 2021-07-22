#!/usr/bin/env python
# coding: utf-8

# # Disease Prediction based on Symtoms

# In[1]:


#Importing Libraries
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import numpy as np
import pandas as pd
#import streamlit as st
import os


# In[2]:


#pip install --upgrade --user pyqt5==5.12


# In[3]:


#pip install --upgrade --user pyqtwebengine==5.12


# In[4]:


#pip install voila


# In[5]:


#List of the symptoms is listed here in list l1.

l1 = ['itching', 'skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
      'crying while touching joints','crying while touching stomach','acidity','crying when head touched',
     'premature birth','crying when back eyes area touched','crying while touching back','crying while touching abdomin',
     'crying while touching chest','crying during bowel movements','crying while touching  knee','crying while touching hip joint',
     'muscle_weakness','muscle_wasting','crying while touching body part','crying while touching belly',
     'irritability','Spinal Injury','red_spots_over_body','poor feeding','skin patches','watering_from_eyes',
     'increased_appetite','frequent urination','crying while touching belly button','coated tongue','hoarse cry',
     'large soft spot (fontanel) on head','nausea','complicated umbilical artery catheter','ulcers_on_tongue',
     'vomiting','painful urination','fatigue','weight_gain','cold_hands_and_feets',
     'weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
     'breathlessness','sweating','dehydration','indigestion','yellowish_skin','dark_urine','loss_of_appetite','constipation',
     'diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','swelled_lymph_nodes','blurred_and_distorted_vision',
     'phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','weakness_in_limbs','fast_heart_rate',
     'bloody_stool','crying while touching neck','dizziness','bruising','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
     'enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','infected mother',
     'stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
     'weakness_of_one_body_side','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases',
     'family_history','mucoid_sputum','rusty_sputum','receiving_blood_transfusion','receiving_unsterile_injections',
     'coma','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','skin_peeling','silver_like_dusting',
     'small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']


# In[6]:


#List of Diseases is listed in list disease.

disease = ['(vertigo) Paroymsal  Positional Vertigo','AIDS','Allergy','Arthritis','Bronchial Asthma',
           'Cervical spondylosis','Chicken pox','Chronic cholestasis','Common Cold','Conjuctivitis',
           'Corona','Dengue','Diabetes','Dimorphic hemmorhoids(piles)','Diaper Rash','Drug Reaction'
           'Gastroenteritis','GERD / Acid Reflux','Heart attack','Hepatitis B','hepatitis A','Hypertension',
           'Paralysis (brain hemorrhage)','Jaundice','Malaria','Typhoid','Hepatitis C','Hepatitis D',
           'Hepatitis E','Tuberculosis','Pneumonia','Varicose veins','Hypothyroidism','Hyperthyroidism'
           'Hypoglycemia','Osteoarthristis','Urinary tract infection','Psoriasis','Impetigo']


# In[7]:


l2=[]
for i in range(0,len(l1)):
    l2.append(0)
#print(l2)


# In[8]:


type(disease)


# In[9]:


#Reading the training .csv file
#df=pd.read_csv("training.csv")
df=pd.read_csv("Training_Prakriti.csv")
#DF= pd.read_csv('training.csv', index_col='prognosis')
DF= pd.read_csv('Training_Prakriti.csv', index_col='prognosis')
#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis':{'(vertigo) Paroymsal  Positional Vertigo':0,'AIDS':1,'Allergy':2,'Arthritis':3,'Bronchial Asthma':4,
    'Cervical spondylosis':5,'Chicken pox':6,'Chronic cholestasis':7,'Common Cold':8,'Conjuctivitis':9,'Corona':10,
    'Dengue':11,'Diabetes ':12, 'Dimorphic hemmorhoids(piles)':13,'Diaper Rash':14,'Drug Reaction':15,'Gastroenteritis':16,'GERD / Acid Reflux':17,'Heart attack':18,'Hepatitis B':19,
    'hepatitis A':20,'Paralysis (brain hemorrhage)':21,'Jaundice':22,'Malaria':23,'Typhoid':24,'Hepatitis C':25,
    'Hepatitis D':26,'Hepatitis E':27,'Tuberculosis':28,'Pneumonia':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Urinary tract infection':35,
    'Psoriasis':36,'Impetigo':37,'Hypertension':38}},inplace=True)

#df.head()
DF.head()


# In[10]:


df.head()


# In[11]:


df.info()
DF.info()


# In[12]:


(df.isnull().sum()).sum()


# In[13]:


df.isnull().sum()


# In[14]:


df.describe()


# In[15]:


X= df[l1]
y = df[["prognosis"]]
np.ravel(y)
X.shape


# In[16]:


'''import seaborn as sns
plt.figure(figsize=(200,100))
sns.heatmap(DF,annot=True)
plt.show()'''


# In[ ]:





# In[17]:


df['prognosis'] = pd.to_numeric(df['prognosis'])
#print(df[pd.to_numeric(df['prognosis'],errors = 'coerce').isnull()])


# In[18]:


df.info()
DF.info()


# In[19]:


len(np.ravel(y))


# In[20]:


#Reading the  testing.csv file
tr=pd.read_csv("Testing_Prakriti.csv")

#Using inbuilt function replace in pandas for replacing the values

tr.replace({'prognosis':{'(vertigo) Paroymsal  Positional Vertigo':0,'AIDS':1,'Allergy':2,'Arthritis':3,'Bronchial Asthma':4,
    'Cervical spondylosis':5,'Chicken pox':6,'Chronic cholestasis':7,'Common Cold':8,'Conjuctivitis':9,'Corona':10,
    'Dengue':11,'Diabetes ':12, 'Dimorphic hemmorhoids(piles)':13,'Diaper Rash':14,'Drug Reaction':15,'Gastroenteritis':16,'GERD / Acid Reflux':17,'Heart attack':18,'Hepatitis B':19,
    'Hepatitis A':20,'Paralysis (brain hemorrhage)':21,'Jaundice':22,'Malaria':23,'Typhoid':24,'Hepatitis C':25,
    'Hepatitis D':26,'Hepatitis E':27,'Tuberculosis':28,'Pneumonia':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Urinary tract infection':35,
    'Psoriasis':36,'Impetigo':37,'Hypertension':38}},inplace=True)

#tr.head()


# In[21]:


tr.shape


# In[22]:


X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
#print(X_test)


# In[23]:


X_test.shape


# **To build the precision of the model, we utilized algorithms which are as per the following**
# * Random Forest algorithm

# In[24]:


#list1 = DF['prognosis'].unique()
def scatterplt(disea):
    x = ((DF.loc[disea]).sum())#total sum of symptom reported for given disease
    x.drop(x[x==0].index,inplace=True)#droping symptoms with values 0
    print(x.values)
    y = x.keys()#storing nameof symptoms in y
    print(len(x))
    print(len(y))
    plt.figure(figsize=(35,10))
    plt.title(disea)
    plt.scatter(y,x.values)
    plt.show()
'''
def scatterinp(sym1,sym2,sym3,sym4,sym5):
    
    x = [sym1,sym2,sym3,sym4,sym5]#storing input symptoms in y
    y = [0,0,0,0,0]#creating and giving values to the input symptoms
    if(sym1!='Select Here'):
        y[0]=1
    if(sym2!='Select Here'):
        y[1]=1
    if(sym3!='Select Here'):
        y[2]=1
    if(sym4!='Select Here'):
        y[3]=1
    if(sym5!='Select Here'):
        y[4]=1
    print(x)
    print(y)
    plt.scatter(x,y)
    plt.show()'''


# In[25]:


'''import seaborn as sns
plt.figure(figsize=(200,100))
sns.heatmap(DF.corr(),annot=True)
plt.show()'''


# # Random Forest Algorithm

# In[26]:


root = Tk()
pred2=StringVar()
def randomforest():
    if len(NameEn.get()) == 0:
        pred2.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here")):
        pred2.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf4 = RandomForestClassifier(n_estimators=100)#This is the number of trees (in general the number of samples on which this algorithm will work then it will aggregate them to give you the final answer) you want to build before taking the maximum voting or averages of predictions
        clf4 = clf4.fit(X,np.ravel(y))
        print("clf4")

        # calculating accuracy 
        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        y_pred=clf4.predict(X_test)
        print("Random Forest")
        print("Accuracy")#TP+TN/Tot
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        print("Confusion matrix")#is it relevant here
        #conf_matrix=confusion_matrix(y_test,y_pred)
        #print(conf_matrix)
    
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = clf4.predict(inputtest)
        predicted=predict[0]#picking 1st pred

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            pred2.set(" ")
            pred2.set(disease[a])
        else:
            pred2.set(" ")
            pred2.set("Not Found")
        #printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred2.get())


# # Building Graphical User Interface

# In[27]:


#Tk class is used to create a root window
root.configure(background='Ivory')
root.title('Optum Care')
root.resizable(0,0)
#xroot.state('zoomed')
#root.attributes('-fullscreen', True)


# In[28]:


Symptom1 = StringVar(root)
Symptom1.set("Select Here")

Symptom2 = StringVar(root)
Symptom2.set("Select Here")

Symptom3 = StringVar(root)
Symptom3.set("Select Here")

Symptom4 = StringVar(root)
Symptom4.set("Select Here")

Symptom5 = StringVar(root)
Symptom5.set("Select Here")
Name = StringVar()
Pin = StringVar()


# In[29]:


prev_win=None
def Reset():
    global prev_win

    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    Symptom5.set("Select Here")
    NameEn.delete(first=0,last=100)
    pred2.set(" ")
    try:
        prev_win.destroy()
        print("Try")
        prev_win=None
    except AttributeError:
        print("except")
        pass


# In[30]:


from tkinter import messagebox
def Exit():
    qExit=messagebox.askyesno("System","Do you want to exit the system")
    
    if qExit:
        root.destroy()
        exit()


# In[31]:


#Headings for the GUI written at the top of GUI
w2 = Label(root, justify = RIGHT, text="Neo Natal Care", fg="Dark Orange", bg="Ivory")#display box where you can put any text or image which can be updated any time as per the code
w2.config(font=("Arial",30,"bold"))#not found
w2.grid(row=1, column=0, columnspan=2, padx=100)#geometry manager organises widgets in a table-like structure in the parent widget
img = PhotoImage(file="optum.png")
imglabel = Label(root,image = img, justify = RIGHT)
imglabel.grid(row=1, column=3)


# In[32]:


#Label for the name
NameLb = Label(root, text="Name of the Patient *", fg="Dark Orange", bg="Ivory")
NameLb.config(font=("Arial",15,"bold "))
NameLb.grid(row=6, column=0, pady=15, sticky=W)


# In[33]:


#Creating Labels for the symtoms
S1Lb = Label(root, text="Symptom 1 *", fg="Dark Orange", bg="Ivory")
S1Lb.config(font=("Arial",15,"bold"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2 *", fg="Dark Orange", bg="Ivory")
S2Lb.config(font=("Arial",15,"bold"))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Dark Orange", bg="Ivory")
S3Lb.config(font=("Arial",15,"bold"))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="Dark Orange", bg="Ivory")
S4Lb.config(font=("Arial",15,"bold"))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="Dark Orange", bg="Ivory")
S5Lb.config(font=("Arial",15,"bold"))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

PinLb = Label(root, text="Pin Code", fg="Dark Orange", bg="Ivory")
PinLb.config(font=("Arial",15,"bold"))
PinLb.grid(row=12, column=0, pady=10, sticky=W)


# In[34]:


#Labels for the different algorithms

destreeLb = Label(root, text="Random Forest", fg="Dark Orange", bg="Ivory")#, width = 20)
destreeLb.config(font=("Arial",15,"bold"))
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

OPTIONS = sorted(l1)


# In[35]:


#Taking name as input from user
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

#Taking Symptoms as input from the dropdown from the user
S1 = (ttk.Combobox)(root,value =OPTIONS,textvariable=Symptom1,width = 30) #dropdown or popup menu that displays a group of objects on a click or keyboard event and lets the user select one option at a time
S1.bind("<<ComboboxSelected>>")
S1.grid(row=7, column=1)

S2 = (ttk.Combobox)(root,value =OPTIONS, textvariable=Symptom2,width = 30)
S2.bind("<<ComboboxSelected>>")
S2.grid(row=8, column=1)

S3 = (ttk.Combobox)(root,value =OPTIONS, textvariable=Symptom3,width = 30)
S3.bind("<<ComboboxSelected>>")
S3.grid(row=9, column=1)

S4 = (ttk.Combobox)(root,value =OPTIONS,textvariable=Symptom4,width = 30)
S4.bind("<<ComboboxSelected>>")
S4.grid(row=10, column=1)

S5 = (ttk.Combobox)(root,value =OPTIONS,textvariable=Symptom5,width = 30)
S5.bind("<<ComboboxSelected>>")
S5.grid(row=11, column=1)

PinEn = Entry(root, textvariable=Pin)
PinEn.grid(row=12, column=1)


# In[36]:


def doctor():
    drs = messagebox.showinfo("Care Details","John : +01919416965034         Hari : +017453298075490        Michelle : +659260427986")
    if drs:
        print("drsdrs")


# In[37]:


#Buttons for predicting the disease using different algorithms

rs = Button(root,text="Reset Inputs", command=Reset,fg="Dark Orange", bg="Ivory")
rs.config(font=("Arial",15,"bold"))
rs.grid(row=13,column=3,padx=10)

rnf = Button(root,text="Prediction", command=randomforest,fg="Dark Orange", bg="Ivory")#used to add buttons
rnf.config(font=("Arial",15,"bold"))
rnf.grid(row=11, column=3,padx=10)

ex = Button(root,text="Exit System", command=Exit,fg="Dark Orange", bg="Ivory")
ex.config(font=("Arial",15,"bold"))
ex.grid(row=17,column=3,padx=10)

dr = Button(root,text="Care Details", command=doctor,fg="Dark Orange", bg="Ivory")
dr.config(font=("Arial",15,"bold"))
dr.grid(row=15,column=3,padx=10)


# In[38]:


#Showing the output of different algorithms

t2=Label(root,font=("Arial",15,"bold"),text="Random Forest",height=1,bg="Ivory"
         ,width=40,fg="Dark Orange",textvariable=pred2,relief="sunken").grid(row=17, column=1, padx=10)


# In[39]:


#calling this function because the application is ready to run
root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




