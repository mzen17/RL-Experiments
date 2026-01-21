import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class System():
    def __init__(self):
        self.x_loc = random.randint(0, 10)
        self.y_loc = random.randint(0, 10)

        self.history = []
        self.path = [(self.x_loc, self.y_loc)]


    def iterate(self, move):
        self.history.append(float(move))
        if move == 0:
            self.x_loc += 1
        elif move == 1:
            self.y_loc += 1
        elif move == 2:
            self.x_loc -= 1
        elif move == 3:
            self.y_loc -= 1
        
        self.path.append((self.x_loc, self.y_loc))

        if (self.x_loc > 20):
            return -10
        elif (self.y_loc > 20):
            return -10
        elif (self.x_loc == -1):
            return 10
        elif (self.y_loc <= -1):
            return -10
        else:
            return -1

class Bot(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 4)
    
    def forward(self, input):
        self.output = self.fc1(input)
        self.ans = torch.softmax(self.output, dim = 0)
        return self.ans
    
def train(system, bot, optimizer):
    saved_probs = []
    rewards = []
    
    while True:
        recent_history = system.history[-30:] if len(system.history) > 30 else system.history
        input_tensor = torch.tensor([float(system.x_loc), float(system.y_loc)] + 
                                   recent_history + 
                                   [-2.0] * (30 - len(recent_history)))
        
        probs = bot.forward(input_tensor)

        m = torch.distributions.Categorical(probs)
        action = m.sample()

        saved_probs.append(m.log_prob(action))

        out = system.iterate(action)
        rewards.append(out)

        if out == -10 or out == 10:
            break
    
    loss = 0
    total_reward = sum(rewards)

    for s in saved_probs:
        loss += -s * total_reward

    print(f"Step Loss: {loss} | Total Reward: {total_reward}" )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return total_reward

def visualize_trajectory(system, episode_num):
    path = system.path
    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]

    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', markersize=4, alpha=0.6)
    plt.plot(x_coords[0], y_coords[0], 'go', label='Start')  # Green dot for start
    plt.plot(x_coords[-1], y_coords[-1], 'ro', label='End')   # Red dot for end
    
    # Draw boundaries
    plt.axvline(x=-1, color='r', linestyle='--', label='Goal (X=-1)')
    plt.axvline(x=20, color='gray', linestyle='--')
    plt.axhline(y=20, color='gray', linestyle='--')
    plt.axhline(y=-1, color='gray', linestyle='--')

    plt.xlim(-5, 25)
    plt.ylim(-5, 25)
    plt.title(f"Trajectory - Episode {episode_num}")
    plt.xlabel("X Location")
    plt.ylabel("Y Location")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"trajectory_ep_{episode_num}.png")
    plt.close()

bot = Bot()
optimizer = torch.optim.Adam(bot.parameters(), lr=0.01)

for i in range(0, 32):
    system = System()
    train(system, bot, optimizer)
    
    if i % 10 == 0 or i == 31:
        print(f"Visualizing Episode {i}...")
        visualize_trajectory(system, i)
    visualize_trajectory(system, i)

