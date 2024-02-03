import streamlit as st
import pickle

@st.cache_resource
def load_model():
    return pickle.load(open('iris_model.pkl','rb'))

st.title('Iris Flower Classification')
st.write("Iris flower classification with Support Vector Machine(SVM) algorithm")
col1,col2 = st.columns(spec=[1,1])
con1 = col1.container(border=True)
in1 = con1.number_input(label="enter sepal length",min_value=0.0,value=1.0,step=.1)
in2 = con1.number_input(label="enter sepal width",min_value=0.0,value=1.0,step=.1)
in3 = con1.number_input(label="enter petal length",min_value=0.0,value=1.0,step=.1)
in4 = con1.number_input(label="enter petal width",min_value=0.0,value=1.0,step=.1)

clicked = st.button(label="Predict")

con2 = col2.container(border=True)
if clicked:
    model = load_model()
    iris_type = model.predict([[in1,in2,in3,in4]])
    con2.write('The flower belongs to the species:')
    if iris_type[0]==0:
        con2.write('**Setosa**')
    else:
        con2.write('**Versicolor**')

