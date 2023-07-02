from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pyomo.environ as pyo
import omlt
from omlt import OmltBlock
from omlt.neuralnet import *
from omlt.neuralnet.activations import ComplementarityReLUActivation
from omlt.io.keras import keras_reader

df = pd.read_csv("sin_quadratic.csv",index_col=[0])

#retrieve input 'x' and output 'y' from the dataframe
x = df["x"]
y = df["y"]


#calculate mean and standard deviation, add scaled 'x' and scaled 'y' to the dataframe
mean_data = df.mean(axis=0)
std_data = df.std(axis=0)
df["x_scaled"] = (df['x'] - mean_data['x']) / std_data['x']
df["y_scaled"] = (df['y'] - mean_data['y']) / std_data['y']

nn4 = load_model('nn4.h5')

#nn4
y_predict_scaled_leaky = nn4.predict(x=df['x_scaled'])
y_predict_leaky = y_predict_scaled_leaky*(std_data['y']) + mean_data['y']

#create a single plot with the original data and each neural network's predictions
fig,ax = plt.subplots(1,figsize = (8,8))
ax.plot(x,y,linewidth = 3.0,label = "data", alpha = 0.5)
ax.plot(x,y_predict_leaky,linewidth = 3.0,linestyle="dotted",label = "leaky-relu")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

#create an omlt scaling object
scaler = omlt.scaling.OffsetScaling(offset_inputs=[mean_data['x']],
                    factor_inputs=[std_data['x']],
                    offset_outputs=[mean_data['y']],
                    factor_outputs=[std_data['y']])

#create the input bounds. note that the key `0` corresponds to input `0` and that we also scale the input bounds
input_bounds={0:((min(df['x']) - mean_data['x'])/std_data['x'],
                 (max(df['x']) - mean_data['x'])/std_data['x'])};
print(scaler)
print("Scaled input bounds: ",input_bounds)

net_relu_leaky = keras_reader.load_keras_sequential(nn4,scaler,input_bounds)

model4_bigm = pyo.ConcreteModel()
model4_bigm.x = pyo.Var(initialize = 0)
model4_bigm.y = pyo.Var(initialize = 0)
model4_bigm.obj = pyo.Objective(expr=(model4_bigm.y))
model4_bigm.nn = OmltBlock()

formulation4_bigm = ReluBigMFormulation(net_relu_leaky)
model4_bigm.nn.build_formulation(formulation4_bigm)

@model4_bigm.Constraint()
def connect_inputs(mdl):
    return mdl.x == mdl.nn.inputs[0]

@model4_bigm.Constraint()
def connect_outputs(mdl):
    return mdl.y == mdl.nn.outputs[0]

status_4_bigm = pyo.SolverFactory('gams:cbc').solve(model4_bigm, tee=False)
solution_4_bigm = (pyo.value(model4_bigm.x),pyo.value(model4_bigm.y))


#print out model size and solution values
print("ReLU BigM Solution:")
print("# of variables: ",model4_bigm.nvariables())
print("# of constraints: ",model4_bigm.nconstraints())
print("x = ", solution_4_bigm[0])
print("y = ", solution_4_bigm[1])

#nn4 - leaky relu 
plt.plot(x,y_predict_leaky,linewidth = 3.0,linestyle="dotted",color = "green")
plt.title("leaky-relu")
plt.scatter([solution_4_bigm[0]],[solution_4_bigm[1]],color = "blue",s = 300, label="bigm")
plt.legend()
plt.show()