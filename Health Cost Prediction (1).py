#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('Health.csv')


# In[3]:


data .head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


print('Number of Rows',data.shape[0])
print('Number of columns',data.shape[0])


# In[7]:


data.info()


# In[8]:


data.isnull()


# In[9]:


data.isnull().sum()


# In[10]:


data.describe(include="all")


# In[11]:


data.head()


# In[12]:


data['sex'].unique()


# In[13]:


data['sex']=data['sex'].map({'female':0,"male":1})


# In[14]:


data.head()


# In[15]:


data['smoker']=data['smoker'].map({'yes':1,'no':0})


# In[16]:


data.head()


# In[17]:


data['region'].unique()


# In[18]:


data['region']=data['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})


# In[19]:


data.head()


# In[20]:


data.columns


# In[21]:


x=data.drop(['charges'],axis=1)


# In[22]:


x


# In[23]:


y=data['charges']


# In[24]:


y


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[27]:


y_train


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[29]:


lr = LinearRegression()
lr.fit(x_train,y_train)
svm=SVR()
svm.fit(x_train,y_train)
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
gr=GradientBoostingRegressor()
gr.fit(x_train,y_train)


# In[30]:


y_pred1 = lr.predict(x_test)
y_pred2 = svm.predict(x_test)
y_pred3 = rf.predict(x_test)
y_pred4 = gr.predict(x_test)

df1 = pd.DataFrame({'Actual':y_test,'lr':y_pred1,'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})


# In[31]:


df1


# In[32]:


import matplotlib.pyplot as plt


# In[33]:


plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['lr'].iloc[0:11],label='lr')
plt.legend()

plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['svm'].iloc[0:11],label='svm')
plt.legend()

plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['rf'].iloc[0:11],label='rf')
plt.legend()

plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['gr'].iloc[0:11],label='gr')

plt.tight_layout()

plt.legend()


# In[34]:


from sklearn import metrics


# In[35]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)


# In[36]:


print(score1,score2,score3,score4)


# In[37]:


s1 = metrics.mean_absolute_error(y_test,y_pred1)
s2 = metrics.mean_absolute_error(y_test,y_pred2)
s3 = metrics.mean_absolute_error(y_test,y_pred3)
s4 = metrics.mean_absolute_error(y_test,y_pred4)


# In[38]:


print(s1,s2,s3,s4)


# Predict Charges

# In[39]:


data = {'age':40,
       'sex':1,
       'bmi':40.30,
       'children':4,
       'smoker':1,
       'region':2}

df = pd.DataFrame(data,index=[0])
df


# In[40]:


new_pred = gr.predict(df)
print(new_pred)


# In[41]:


gr=GradientBoostingRegressor()
gr.fit(x,y)


# In[42]:


import joblib


# In[43]:


joblib.dump(gr,'model_joblib_gr')


# In[44]:


model = joblib.load('model_joblib_gr')


# In[ ]:





# In[45]:


model.predict(df)


# GUI

# In[46]:


from tkinter import *


# In[47]:


import joblib


# In[48]:


def show_entry() :
    
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())

    model = joblib.load('model_joblib_gr')
    result=model.predict([[p1,p2,p3,p4,p5,p6]])
    
    Label(master, text = "Health cost").grid(row=7)
    Label(master, text=result).grid(row=8)
    

master=Tk()
master.title("Health Cost Prediction")
label = Label(master, text ="Health cost prediction", bg="black",fg="white").grid(row=0,columnspan=2)

Label(master, text = "ENter Your Age").grid(row=1)
Label(master, text = "Male or Female [1/0]").grid(row=2)
Label(master, text = "ENter Your BMI Value").grid(row=3)
Label(master, text = "ENter Number of children ").grid(row=4)
Label(master, text = "smoker Yes/No [1/0]").grid(row=5)
Label(master, text = "region [1-4]").grid(row=6)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)


e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)

Button(master,text="Predict",command=show_entry).grid()

mainloop()


# In[49]:


import streamlit as st
import joblib

def main():
    html_temp = """
    <dev style="background-colorlightblue;padding:16px">
    <h2 style="color:black";text-align:center> Health Cost Prediction using ML</h2>
    </div>
    
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # load the model
    model = joblib.load('model_joblib_gr')
    
    p1 = st.slider("Enter Your Age",18,100)
    
    s1=st.selectbox("Sex",("Male","Female"))
    if s1=="Male":
        p2=1
    else:
        p2=0

    p3 =st.number_input("Enter Your BMI Value")
    p4 = st.slider("Enter Number of Children",0,4) 
    
    s2=st.selectbox("Smoker",("Yes","No"))
    if s2=="Yes":
        p5=1
    else:
        p5=0
        
    p6 = st.slider("Enter Your Region [1-4]",1,4)
    
    if st.button('Predict'):
        prediction = model.predict([[p1,p2,p3,p4,p5,p6]])
        st.balloons()
        st.success('Insurance Amount is {} '.format(round(prediction[0],2))
  
if __name__ == '__main__':
    main()


# In[ ]:




