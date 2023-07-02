# data manipulation and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# tensorflow objects
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

# pyomo for optimization
import pyomo.environ as pyo

# omlt for interfacing our neural network with pyomo
import omlt
from omlt import OmltBlock
from omlt.neuralnet import *
from omlt.neuralnet.activations import ComplementarityReLUActivation
from omlt.io.keras import keras_reader
import pickle

# set the net name here
netname = 'faulty_nets/leaky_-1_relu_opterror.h5'

# generate training data for the neural network
def generate_training_data(alpha, start, end, step):
    x = np.arange(start, end, step)
    y = np.where(x > 0, x, np.where(x < 0, alpha * x, 0))  # Modify the conditions as needed
    df = pd.DataFrame({'x': x, 'y': y})
    # Retrieve input 'x' and output 'y' from the dataframe
    x = df["x"]
    y = df["y"]

    # Calculate mean and standard deviation, add scaled 'x' and scaled 'y' to the dataframe
    mean_data = df.mean(axis=0)
    std_data = df.std(axis=0)
    df["x_scaled"] = (df['x'] - mean_data['x']) / std_data['x']
    df["y_scaled"] = (df['y'] - mean_data['y']) / std_data['y']
    return df

start = -10
end = 10
step = 1

alp = -1.5
df = generate_training_data(alp,start,end,step)

#retrieve input 'x' and output 'y' from the dataframe
x = df["x"]
y = df["y"]

#calculate mean and standard deviation, add scaled 'x' and scaled 'y' to the dataframe
mean_data = df.mean(axis=0)
std_data = df.std(axis=0)
df["x_scaled"] = (df['x'] - mean_data['x']) / std_data['x']
df["y_scaled"] = (df['y'] - mean_data['y']) / std_data['y']

# load net from h5 file using keras
net = load_model(netname)

#net
y_predict_scaled_leaky = net.predict(x=df['x_scaled'])
y_predict_leaky = y_predict_scaled_leaky*(std_data['y']) + mean_data['y']

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

net_relu_leaky = keras_reader.load_keras_sequential(net,scaler,input_bounds)

model_bigm = pyo.ConcreteModel()
model_bigm.x = pyo.Var(initialize = 0)
model_bigm.y = pyo.Var(initialize = 0)
model_bigm.obj = pyo.Objective(expr=(model_bigm.y))
model_bigm.nn = OmltBlock()

formulation4_bigm = ReluBigMFormulation(net_relu_leaky)
model_bigm.nn.build_formulation(formulation4_bigm)

@model_bigm.Constraint()
def connect_inputs(mdl):
    return mdl.x == mdl.nn.inputs[0]

@model_bigm.Constraint()
def connect_outputs(mdl):
    return mdl.y == mdl.nn.outputs[0]

status_4_bigm = pyo.SolverFactory('gams:cbc').solve(model_bigm, tee=False)
solution_4_bigm = (pyo.value(model_bigm.x),pyo.value(model_bigm.y))

# extract the name of the neural network from netname
netname2 = netname.split(".")[0]

#create a single plot with the original data and each neural network's predictions
fig,ax = plt.subplots(1,figsize = (8,8))
ax.plot(x,y,linewidth = 3.0,label = "data", alpha = 0.5)
ax.plot(x,y_predict_leaky,linewidth = 3.0,linestyle="dotted",label = netname2)
ax.scatter([solution_4_bigm[0]],[solution_4_bigm[1]],color = "blue",s = 300)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
