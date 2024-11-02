import numpy as np
import random

# Environment settings
GRID_SIZE = 3
START = (0, 0)
TREASURE = (2, 2)  # Set the treasure location

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1    # Exploration rate
num_episodes = 300

# Initialize Q-table: each state-action pair has a value initialized to 0
Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # 4 actions: up, down, left, right

# Actions (up, down, left, right)
actions = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

# Check if a position is valid
def is_valid(pos):
    return 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE

# Choose action based on epsilon-greedy strategy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(4))  # explore
    return np.argmax(Q[state[0], state[1]])  # exploit

# Update Q-value
def update_q(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state[0], next_state[1]])
    Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], best_next_action] - Q[state[0], state[1], action])

# Train the agent
for episode in range(num_episodes):
    state = START
    while state != TREASURE:
        action = choose_action(state)
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        
        if not is_valid(next_state):  # invalid move
            reward = -1
            next_state = state  # stay in place
        elif next_state == TREASURE:  # found treasure
            reward = 10
        else:
            reward = -1  # regular move cost
        
        update_q(state, action, reward, next_state)
        state = next_state

# Display learned Q-table
print("Q-Table for Treasure Hunt:\n", Q)
