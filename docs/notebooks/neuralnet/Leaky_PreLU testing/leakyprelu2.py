# data manipulation and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

# tensorflow objects
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
import statistics

# pyomo for optimization
import pyomo.environ as pyo

# omlt for interfacing our neural network with pyomo
import omlt
from omlt import OmltBlock
from omlt.neuralnet import *
from omlt.neuralnet.activations import ComplementarityReLUActivation
from omlt.io.keras import keras_reader


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
alphas = [1.5, 1, 0.5, 0, -0.5, -1, -1.5]

prelu_nets = []
leakynets = []
for alpha in alphas:
    model = Sequential(name=f'leakynet_{alpha}')
    model.add(Input(1))
    model.add(Dense(1, activation=LeakyReLU(alpha=alpha), kernel_initializer='random_uniform', bias_initializer='random_uniform'))
    model.add(Dense(1, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
    model.compile(optimizer=Adam(), loss='mse')
    leakynets.append(model)

for model in leakynets:
    alpha = model.layers[0].get_config()['activation']['config']['alpha']
    print(f'Training leakynet with alpha={alpha}...')
    df = generate_training_data(alpha, start, end, step)
    model.fit(x=df['x_scaled'], y=df["y_scaled"], epochs=10, batch_size=32)  # Adjust the number of epochs and batch size as needed

for model in leakynets:
    alpha = model.layers[0].get_config()['activation']['config']['alpha']
    print(f"Alpha value for {model.name}: {alpha}")

prelu_nets = []
for alpha in alphas:
    model = Sequential(name=f'prelu_net_{alpha}')
    model.add(Input(1))
    alpha_initializer1 = initializers.Constant(value=alpha)
    model.add(Dense(1, activation=PReLU(alpha_initializer=alpha_initializer1), kernel_initializer='random_uniform', bias_initializer='random_uniform'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    prelu_nets.append(model)

for model in prelu_nets:
    alpha = model.layers[0].get_config()['activation']['config']['alpha_initializer']['config']['value']
    print(f'Training PReLU net with alpha={alpha}...')
    df = generate_training_data(alpha, start, end, step)
    model.fit(x=df['x_scaled'], y=df["y_scaled"], epochs=10, batch_size=32)  # Adjust the number of epochs and batch size as needed

for model in prelu_nets:
    alpha = model.layers[0].get_config()['activation']['config']['alpha_initializer']['config']['value']
    print(f"Alpha value for {model.name}: {alpha}")

# Pickle the prelu_nets list
with open('prelu_nets.pkl', 'wb') as f:
    pickle.dump(prelu_nets, f)

# Pickle the leakynets list
with open('leakynets.pkl', 'wb') as f:
    pickle.dump(leakynets, f)

# Unpickle the prelu_nets list
with open('prelu_nets.pkl', 'rb') as f:
    prelu_nets = pickle.load(f)

# Unpickle the leakynets list
with open('leakynets.pkl', 'rb') as f:
    leakynets = pickle.load(f)

# function to evaluate a point on the neuralnet using OMLT
def OMLTcheck(net, data):
    df = data

    mean_data = df.mean(axis=0)
    std_data = df.std(axis=0)

    df["x_scaled"] = (df['x'] - mean_data['x']) / std_data['x']
    df["y_scaled"] = (df['y'] - mean_data['y']) / std_data['y']
    #create an omlt scaling object
    scaler = omlt.scaling.OffsetScaling(offset_inputs=[mean_data['x']],
                        factor_inputs=[std_data['x']],
                        offset_outputs=[mean_data['y']],
                        factor_outputs=[std_data['y']])

    #create the input bounds. note that the key `0` corresponds to input `0` and that we also scale the input bounds
    input_bounds={0:((min(df['x']) - mean_data['x'])/std_data['x'],
                    (max(df['x']) - mean_data['x'])/std_data['x'])};
    # print(scaler)
    # print("Scaled input bounds: ",input_bounds)

    y_predict_scaled = net.predict(x=df['x_scaled'])
    y_predict = y_predict_scaled*(std_data['y']) + mean_data['y']

    net_relu = keras_reader.load_keras_sequential(net, scaler, input_bounds)

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

    #create a single plot with each neural network's predictions
    fig,ax = plt.subplots(1,figsize = (8,8))
    ax.plot(df['x'], y_predict, linestyle='', marker='o', color='green', label='net/data outputs')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    #print out model size and solution values
    print("ReLU BigM Solution:")
    print("# of variables: ",model_bigm.nvariables())
    print("# of constraints: ",model_bigm.nconstraints())
    print("x = ", solution_bigm[0])
    print("y = ", solution_bigm[1])

    #net
    plt.title("OMLT vs Keras values"+ " "+net.name)
    plt.scatter([solution_bigm[0]],[solution_bigm[1]],color = "blue",s = 300, label="OMLT sol")
    plt.legend()
    plt.show()
    plt.close()

    # print(net.name)
    # print(y_predict.min())
    # print(y_predict.min()-solution_bigm[1])
    # tolerance = 1.5e-5
    # assert np.isclose(y_predict.min(), solution_bigm[1], atol=tolerance)
    # min_index = np.argmin(y_predict)
    # value = df['x'][min_index]
    # print(df['x'][min_index])
    # assert np.isclose(value, solution_bigm[0], atol=tolerance)

#Test PReLU
for n in range(0,7):
    alpha = prelu_nets[0].layers[0].get_config()['activation']['config']['alpha_initializer']['config']['value']
    df = generate_training_data(alpha, start, end, step)
    OMLTcheck(prelu_nets[n], df)

#Test LeakyReLU
for n in range(0,7):
    alpha = leakynets[n].layers[0].get_config()['activation']['config']['alpha']
    inputs, outputs = generate_training_data(alpha, start, end, step)
    OMLTcheck(leakynets[n], inputs,outputs)