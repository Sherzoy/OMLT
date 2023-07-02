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

alphas = [1.5, 1, 0.5, 0, -0.5, -1, -1.5]
leakynets = []

for alpha in alphas:
    model = Sequential(name=f'leakynet_{alpha}')
    model.add(Input(1))
    model.add(Dense(1, activation=LeakyReLU(alpha=alpha), kernel_initializer='random_uniform', bias_initializer='random_uniform'))
    model.add(Dense(1, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
    model.compile(optimizer=Adam(), loss='mse')
    leakynets.append(model)

alphas = [1.5, 1, 0.5, 0, -0.5, -1, -1.5]
prelu_nets = []

for alpha in alphas:
    model = Sequential(name=f'prelu_net_{alpha}')
    model.add(Input(1))
    alpha_initializer1 = initializers.Constant(value=alpha)
    model.add(Dense(1, activation=PReLU(alpha_initializer=alpha_initializer1), kernel_initializer='random_uniform', bias_initializer='random_uniform'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    prelu_nets.append(model)

# Input data
inputs = np.array([1, 2, 3, 4, 5])

# Create empty arrays to store outputs
leaky_outputs = []
prelu_outputs = []

# Feed inputs to each neural network and store their outputs
for model in leakynets:
    outputs = model.predict(inputs)
    leaky_outputs.append(outputs.flatten())

for model in prelu_nets:
    outputs = model.predict(inputs)
    prelu_outputs.append(outputs.flatten())

# function to evaluate a point on the neuralnet using OMLT
def OMLTcheck(net, inputs, outputs):

    # Create a dictionary with the field names and corresponding arrays
    data = {'x': inputs, 'y': outputs.flatten()}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    #retrieve input 'x' and output 'y' from the dataframe
    x = df["x"]
    y = df["y"]

    y_predict = net.predict(x=df['x'])

    #create a single plot with the original data and each neural network's predictions
    fig,ax = plt.subplots(1,figsize = (8,8))
    ax.plot(x, y_predict, linestyle='', marker='o', color='green', label='net/data outputs')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    #create the input bounds. note that the key `0` corresponds to input `0` and that we also scale the input bounds
    input_bounds={0:((min(df['x']) - 0)/1,
                 (max(df['x']) - 0)/1)}
    print(input_bounds)
    net_relu = keras_reader.load_keras_sequential(net, unscaled_input_bounds=input_bounds)

    model_bigm = pyo.ConcreteModel()
    model_bigm.x = pyo.Var(initialize =1)
    model_bigm.y = pyo.Var(initialize = 0)
    model_bigm.obj = pyo.Objective(expr=(model_bigm.y))
    model_bigm.nn = OmltBlock()

    formulation_bigm = ReluBigMFormulation(net_relu)
    model_bigm.nn.build_formulation(formulation_bigm)

    @model_bigm.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.nn.inputs[0]

    @model_bigm.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.nn.outputs[0]

    status_bigm = pyo.SolverFactory('gams:cbc').solve(model_bigm, tee=True)
    solution_bigm = (pyo.value(model_bigm.x),pyo.value(model_bigm.y))

    #print out model size and solution values
    print("ReLU BigM Solution:")
    print("# of variables: ",model_bigm.nvariables())
    print("# of constraints: ",model_bigm.nconstraints())
    print("x = ", solution_bigm[0])
    print("y = ", solution_bigm[1])

    #net - parametric relu 
    plt.title("OMLT vs Keras values")
    plt.scatter([solution_bigm[0]],[solution_bigm[1]],color = "blue",s = 300, label="OMLT sol")
    plt.legend()
    plt.show()
    plt.close()

n = 5
layer = prelu_nets[n].layers[1]
print(layer.get_weights())
alphalayer = layer.get_weights()[0]
weights = layer.get_weights()[1]
biases = layer.get_weights()[2]
print("inputs",inputs)
print("weights",prelu_outputs[n])
print("biases", weights)
print("alpha", alphalayer)
OMLTcheck(prelu_nets[n],inputs,prelu_outputs[n])

# OMLTcheck(leakynets[n],inputs,leaky_outputs[n])
#(0,5)
