# Self-Driving Car with Deep Q-Learning

This code implements a simplified self-driving car simulation that uses the Deep Q-Learning algorithm to navigate.

## Requirements
- numpy
- random
- os
- torch
## Neural Network Architecture
The Network class defines the neural network architecture. It consists of two fully connected layers. The network is trained to predict the Q-values for each action given the state.

## Experience Replay
To enhance the learning process, we store the agent's experiences and later sample from this storage to train the neural network. This method is called Experience Replay. The ReplayMemory class provides the functionality for storing and sampling experiences.

## Deep Q-Learning Implementation
The Dqn class is where the Deep Q-Learning magic happens. This class initializes the neural network, memory storage, optimizer, and other required components. It also contains methods to:

- Select an action (select_action)
- Learn from the experiences (learn)
- Update the Q-values (update)
- Save and Load the model weights (save and load)
  
## Self-Driving Car Simulation
The code then integrates the self-driving car logic using the Kivy framework for visualization.

- The Car class represents the car, its sensors, and the logic to move it.
- The Game class simulates the environment, which updates the car's position, checks for sand or road, and provides rewards based on the car's actions.
- The MyPaintWidget class provides painting tools to simulate sand on the road, representing obstacles.

## Running the Simulation
To run the simulation:

- Set up the environment, making sure you have all the required libraries installed.
- Run the "map.py" script. A window will pop up, simulating the car's environment.
- You can draw barriers using your mouse. The car will try to avoid the barriers while navigating.
