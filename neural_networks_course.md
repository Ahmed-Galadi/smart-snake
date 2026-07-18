# 🧠 Neural Networks: From Zero to Deep Q-Learning

> A practical, hands-on course built alongside your Snake AI project.
> *Last updated: July 2026*

---

## Table of Contents

1. [What is a Neural Network?](#1-what-is-a-neural-network)
2. [The Neuron: The Building Block](#2-the-neuron-the-building-block)
3. [How Information Flows: Forward Propagation](#3-how-information-flows-forward-propagation)
4. [Activation Functions: Why We Need Non-Linearity](#4-activation-functions-why-we-need-non-linearity)
5. [Layers: Depth and Width](#5-layers-depth-and-width)
6. [How Neural Networks Learn: Backpropagation](#6-how-neural-networks-learn-backpropagation)
7. [Gradient Descent: The Optimization Engine](#7-gradient-descent-the-optimization-engine)
8. [The Loss Function: Measuring Error](#8-the-loss-function-measuring-error)
9. [Putting It All Together: The Training Loop](#9-putting-it-all-together-the-training-loop)
10. [From Neural Networks to Deep Q-Networks (DQN)](#10-from-neural-networks-to-deep-q-networks-dqn)
11. [Advanced Concepts in Your Snake Agent](#11-advanced-concepts-in-your-snake-agent)
12. [Glossary](#12-glossary)
13. [Further Reading](#13-further-reading)

---

## 1. What is a Neural Network?

### The Big Picture

A **neural network** is a computing system that learns to map inputs to outputs by example. Think of it as a **universal function approximator** — given enough data and enough neurons, it can theoretically learn any mathematical function.

### Real-World Analogy: The Chef

Imagine teaching a chef to cook. You don't give them a recipe book (traditional programming). Instead:
1. You show them ingredients **(input)**
2. You tell them what the dish should taste like **(target output)**
3. They try, mess up, learn, and adjust **(training)**
4. Over time, they get better **(learning)**

The neural network does exactly this, but with math.

### The Three Pillars of a Neural Network

```
┌─────────────────────────────────────────────────────┐
│                  NEURAL NETWORK                      │
├─────────────────────────────────────────────────────┤
│  1. Architecture    2. Learning Algorithm   3. Data  │
│  (structure)        (how it improves)       (fuel)  │
└─────────────────────────────────────────────────────┘
```

### Why "Neural"?

The name comes from biological inspiration:

| Biological Brain | Artificial Neural Network |
|-----------------|--------------------------|
| Neuron (brain cell) | Node / perceptron |
| Synapse (connection) | Weight (connection strength) |
| Firing threshold | Activation function |
| Learning through repetition | Training through backpropagation |

**Key insight:** Real brains are far more complex. We're not simulating brains — we're *inspired* by them.

---

## 2. The Neuron: The Building Block

### The Mathematical Neuron

A single neuron is surprisingly simple:

```
    x₁ ──── w₁ ──┐
                  │
    x₂ ──── w₂ ──┼──→ Σ (sum) → f(Σ) → output
                  │
    x₃ ──── w₃ ──┘
                  │
                 1 ─── b ──┘  (bias)
```

### The Formula

```
output = activation( w₁x₁ + w₂x₂ + w₃x₃ + b )
```

Where:
- **x₁, x₂, x₃** = input values (features)
- **w₁, w₂, w₃** = weights (how important each input is)
- **b** = bias (the neuron's "threshold")
- **Σ** = weighted sum (linear combination)
- **activation()** = activation function (adds non-linearity)

### Code Example: One Neuron

```python
import math

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights randomly (small values)
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
    
    def forward(self, inputs):
        # Step 1: Weighted sum
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        
        # Step 2: Apply activation function
        output = self.sigmoid(z)
        return output
    
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
```

### Visual Intuition

Think of weights as **volume knobs**:
- **Large positive weight** → this input strongly influences the output
- **Large negative weight** → this input suppresses the output
- **Zero weight** → this input is ignored

The bias is like the neuron's **baseline excitement** — how willing it is to fire even without strong inputs.

---

## 3. How Information Flows: Forward Propagation

### The forward pass is how input becomes output.

```
                        ┌───────────────────┐
    Input Layer         │   Hidden Layer    │    Output Layer
    (features)          │   (learned reps)  │    (predictions)
                        │                   │
    danger_straight ───→│                   │
    danger_left     ───→│    ┌──○──┐       │──→ action 0 (up)
    danger_right    ───→│    │     │       │──→ action 1 (down)
    dir_up          ───→│    │  ○  │       │──→ action 2 (left)
    dir_down        ───→│    │     │       │──→ action 3 (right)
    dir_left        ───→│    └──○──┘       │
    dir_right       ───→│                   │
    apple_up        ───→│     256           │
    apple_down      ───→│    neurons        │
    apple_left      ───→│                   │
    apple_right     ───→│                   │
    ...             ───→│                   │
                        └───────────────────┘
```

### Step-by-Step Through Your Snake Network

```python
# Your actual network (from dqn_agent.py)
self.net = nn.Sequential(
    nn.Linear(14, 256),      # Layer 1: 14 inputs → 256 neurons
    nn.ReLU(),                # Activation: non-linearity!
    nn.Linear(256, 256),      # Layer 2: 256 → 256 (deep layer)
    nn.ReLU(),                # Activation again
    nn.Linear(256, 4)         # Layer 3: 256 → 4 actions
)
```

**What happens when the snake sees danger on the right:**

1. **Input** `[0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0.8, 0.3, 0.6]`
   - danger = [straight=0, left=0, right=1] ← wall on right!
   - direction = up
   - apple = [up, down, left, right]
   - free_space = [straight=0.8, left=0.3, right=0.6]

2. **Layer 1** (`Linear(14, 256)`): Each of the 256 neurons looks at all 14 inputs, weighted differently. Some will "light up" for danger, others for apple position.

3. **ReLU**: Kills negative values (neurons that aren't firing). Keeps positive values. This is what creates **sparse, meaningful representations**.

4. **Layer 2** (`Linear(256, 256)`): Second layer of 256 neurons looks at the patterns from Layer 1. Now it can detect **combinations of features** like "danger on the right AND apple is left."

5. **Layer 3** (`Linear(256, 4)`): Final layer condenses everything into 4 scores — how good each action is.

6. **Output** `[2.3, -0.5, 1.1, 3.8]`
   - Up: 2.3
   - Down: -0.5
   - Left: 1.1
   - **Right: 3.8** ← highest! But wait — danger is right! The network is wrong.
   - *This is why we train.*

### The Math of Forward Propagation (Matrix Form)

What looks complex as a diagram is elegant in matrix math:

```
z₁ = W₁ · x + b₁        # Layer 1 linear transformation
a₁ = ReLU(z₁)            # Layer 1 activation
z₂ = W₂ · a₁ + b₂       # Layer 2 linear transformation
a₂ = ReLU(z₂)            # Layer 2 activation
z₃ = W₃ · a₂ + b₃       # Layer 3 linear transformation
output = z₃              # No activation on final layer (raw scores)
```

In matrix form:
- **Input** x has shape `[14]` (1D vector of 14 features)
- **W₁** has shape `[256, 14]` (256 neurons, each with 14 weights)
- **Output** has shape `[4]` (4 Q-values for actions)

---

## 4. Activation Functions: Why We Need Non-Linearity

### The Problem with Linear

If we stacked linear layers without activation functions:

```
layer1(x) = W₁·x + b₁       # Still linear
layer2(layer1(x)) = W₂·(W₁·x + b₁) + b₂ = (W₂·W₁)·x + (W₂·b₁ + b₂)
```

This collapses to **a single linear layer!** No matter how many layers you stack, without activation functions, the entire network is equivalent to one linear transformation.

A linear function can only learn **straight lines** — it can't learn patterns like "go left if the apple is left AND there's no wall." We need **curves** (non-linearity) to learn complex patterns.

### Common Activation Functions

#### ReLU (Rectified Linear Unit) — *Used in your network*

```
f(x) = max(0, x)

        |
    ▲   |   /
    |   |  /
    |   | /
    |   |/
────┼───┴───────────
    |  0
```

```python
def relu(x):
    return max(0, x)
```

**Pros:**
- Simple and fast (just max with 0)
- No "vanishing gradient" problem (more on this later)
- Creates sparse activations (many neurons are "off")

**Cons:**
- "Dying ReLU" problem: if a neuron always outputs ≤ 0, its gradient is 0 and it never learns again
- Not zero-centered

#### Sigmoid (Logistic)

```
f(x) = 1 / (1 + e^(-x))

    ▲
  1 |───────○───────
    |      / \
    |     /   \
    |    /     \
    |   /       \
  0 ───○─────────○───
        -5      5
```

**Used for:** Output layer of binary classifiers (output between 0 and 1 = probability)

**Downside:** Suffers from vanishing gradient at extreme values.

#### Tanh (Hyperbolic Tangent)

```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    ▲
  1 |───────○───────
    |      / \
    |     /   \
    |    /     \
    |   /       \
  0 ───○─────────○───
    |  /
 -1 |○───────────────
        -5      5
```

**Used for:** Hidden layers when you want zero-centered activations.

### Why ReLU Won in Practice

1. **Computationally cheap** — just `max(0, x)`, no exponentials
2. **No vanishing gradient** for positive values (gradient = 1)
3. **Better convergence** in practice for deep networks
4. **Sparsity** — many neurons output 0, making the network efficient

---

## 5. Layers: Depth and Width

### Width vs. Depth

```
Shallow & Wide:                    Deep & Narrow:
┌─────────────────┐                ┌────────────┐
│  ○ ○ ○ ○ ○ ○ ○  │                │    ○ ○     │
│  ○ ○ ○ ○ ○ ○ ○  │  1000          │    ○ ○     │
│  ○ ○ ○ ○ ○ ○ ○  │  neurons       │    ○ ○     │
│       ○ ○        │                │    ○ ○     │
└─────────────────┘                │    ○ ○     │
                                    │    ○ ○     │
                                    │    ○ ○     │
                                    │    ○ ○     │  12 layers
                                    │    ○ ○     │  of 8 neurons
                                    │    ○ ○     │
                                    │    ○ ○     │
                                    │    ○ ○     │
                                    │    ○ ○     │
                                    └────────────┘
```

**Wide networks** → learn broader patterns, but may miss subtle hierarchies
**Deep networks** → learn hierarchical features, layer by layer

### The Hierarchy of Learning (Why Depth Matters)

Deep networks learn **compositional features**:

```
Layer 1:        Danger detection (is there a wall? self-collision?)
                └── Simple edge/collision detectors

Layer 2:        Situational awareness
                └── "Danger right AND apple left" → "turn left
                    to avoid wall AND get closer to apple"

Layer 3 (output): Action selection
                  └── Weigh all options → pick best action
```

This hierarchy is why deep learning works so well — each layer builds on the patterns detected by the previous layer.

### Your Network Architecture Deep Dive

```python
# 14 inputs
nn.Linear(14, 256)      #  14 * 256 + 256 = 3,840 parameters
nn.ReLU()
nn.Linear(256, 256)     # 256 * 256 + 256 = 65,792 parameters
nn.ReLU()
nn.Linear(256, 4)       # 256 * 4 + 4 = 1,028 parameters
```

**Total trainable parameters: 70,660**

That seems like a lot, but compare it to modern LLMs which have **trillions** of parameters!

---

## 6. How Neural Networks Learn: Backpropagation

### The Learning Problem

After forward propagation, we have:
- Input → **Prediction** (output)
- We also have the **correct answer** (target from training data)
- The difference is the **error**

**Question:** How do we know which weights to adjust and by how much?

**Answer:** Backpropagation — the chain rule of calculus applied to neural networks.

### Step 1: The Chain Rule (Intuition)

Imagine a simple network:
```
x → f → g → loss
L(x) = g(f(x))
```

If the output is wrong (high loss), how much is f vs. g responsible?

```
∂loss/∂x = ∂loss/∂g · ∂g/∂f · ∂f/∂x
          ↑           ↑          ↑
       error at    error at    error at
       output      layer g     layer f
```

Each layer's responsibility = the product of all the derivatives above it.
This is called the **chain rule** — and it's the mathematical engine behind all deep learning.

### Step 2: Backpropagation Walkthrough

```
Forward Pass (compute predictions):
                                   ∂loss/∂ŷ
Input → Layer 1 → Layer 2 → Layer 3 → ŷ → loss(ŷ, y)
                                            ↑
                                       Compare to target

Backward Pass (compute gradients):
                                    ∂loss/∂w₃ = ∂loss/∂ŷ · ∂ŷ/∂w₃
Input → Layer 1 → Layer 2 → Layer 3 → ŷ → loss(ŷ, y)
           ↑              ↑              ↑
        ∂loss/∂w₁      ∂loss/∂w₂      ∂loss/∂w₃
```

Each weight gets a gradient telling it:
1. **Direction** (increase or decrease the weight?)
2. **Magnitude** (how much to change?)

### Step 3: Gradient Flow Through Your Network

When your snake crashes into a wall:
- The Q-value for the action that caused the crash was too high
- Loss = high (the network was wrong)
- Gradients flow backward from the loss through all 70,660 parameters
- Each weight learns: "I contributed to this mistake, adjust accordingly"

### Code: Backpropagation in PyTorch

```python
# Forward pass
predicted_q_values = self.net(state)    # Forward pass
action = argmax(predicted_q_values)      # Pick best action
reward = env.step(action)                # Take action
target = reward + gamma * next_q_value   # Compute target

# Compute loss (how wrong were we?)
loss = self.loss_fn(predicted_q_values, target)

# Backward pass
self.optimizer.zero_grad()               # Reset gradients from last step
loss.backward()                          # BACKPROPAGATE ← magic happens here
torch.nn.utils.clip_grad_norm_(...)      # Prevent exploding gradients
self.optimizer.step()                    # Update all weights
```

---

## 7. Gradient Descent: The Optimization Engine

### The Core Idea

Gradient descent is the algorithm that **actually updates the weights**. Think of it like hiking down a mountain in fog:

- You can only feel the slope at your feet (the gradient)
- You take a step downhill (update weights)
- Eventually, you reach the valley floor (minimum loss)

### The Update Rule

```
w_new = w_old - learning_rate × gradient
```

### Three Key Components

#### 1. Learning Rate (α = alpha)

Controls **step size**:

```
Too small (α = 0.0001):                     Too large (α = 1.0):
↓ Every step is tiny                        ↓ We overshoot!
  Might get stuck in local minima             Might never converge
  Very slow                                    Diverges

Just right (α = 0.001):
↓ Steady progress
  Converges reliably
```

**In your Snake agent:**
```python
self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
```
Your learning rate is 0.001 (1e-3) — a standard, safe choice.

#### 2. Gradient (∇ = "nabla" = slope)

The gradient tells you:
- **Direction** to move (sign: + or -)
- **Steepness** (magnitude: how big of a change)

```
        Loss surface
            ▲
         ___|___
        /   |   \
       /    |    \
      /     |     \       ← flat = small gradient (almost converged)
     /      |      \
    /       |       \
───○────────┼────────○───→  Weight
    \       |       /
     \      |      /
      \     |     /        ← steep = large gradient (far from optimal)
       \    |    /
        \   |   /
         \__|__/
```

#### 3. Batch Size

How many experiences to learn from at once:

```
Stochastic (batch=1):   Mini-batch (batch=64):    Full batch (all data):
  Noisy learning          Stable learning          Very stable
  Fast per step           Balanced                 Very slow
  May not converge        Most common              Computationally
                          in practice              expensive
```

**In your agent:**
```python
def train(self, batch_size=64, gamma=0.95):
```
You sample 64 random experiences from memory and learn from all of them at once.

### Types of Gradient Descent

| Variant | Description | Trade-off |
|---------|-------------|-----------|
| **SGD** | Stochastic Gradient Descent | Simple but noisy |
| **Momentum** | Adds velocity from previous steps | Smoother, faster convergence |
| **Adam** | Adaptive moments (learning rate per parameter) | **Best default choice** |
| **RMSprop** | Adaptive learning rates | Good for RNNs |

**Your agent uses Adam** — the modern default optimizer that adapts the learning rate for each parameter individually.

### Visualizing the Loss Landscape

```
Loss
▲
|        \                 /        ← Local minimum (bad)
|         \     ___       /
|          \   /   \     /
|           \ /     \   /
|            \       \ /         ← Global minimum (best!)
|             \       /
|              \     /
|               \___/
└───────────────────────────────→ Training iterations
                   ↓
            Loss decreasing over time
```

A good training run shows loss decreasing smoothly. A bad one oscillates wildly or plateaus too early.

---

## 8. The Loss Function: Measuring Error

### Why We Need It

The network doesn't know it's wrong by itself. The loss function quantifies **how wrong** the prediction is, which drives the entire learning process.

### Mean Squared Error (MSE)

**What it does:** Squares the difference between prediction and target. Big errors are punished quadratically.

```
MSE = (1/n) × Σ(predicted - target)²
```

```python
# If prediction = 5, target = 3:
loss = (5 - 3)² = 4    # Moderate error
# If prediction = 10, target = 3:
loss = (10 - 3)² = 49  # HUGE error (penalized much more!)
```

**Used in your Snake agent:**
```python
self.loss_fn = nn.MSELoss()
```

### Why MSE for Q-Learning?

In Q-learning, we want the network to output Q-values (expected future rewards). MSE works well because:
1. Big mistakes get heavily penalized (quadratic) → the network learns quickly from crashes
2. Small errors matter less → the network can be "close enough"
3. The squared error is differentiable everywhere → gradients always exist

### Other Common Loss Functions

| Loss | Formula | Used For |
|------|---------|----------|
| **MSE** | (ŷ - y)² | Regression, Q-learning |
| **MAE** | \|ŷ - y\| | Robust regression |
| **Cross-Entropy** | -y·log(ŷ) | Classification |
| **Huber** | MSE for small errors, MAE for large | Robust regression |

---

## 9. Putting It All Together: The Training Loop

### The Complete Cycle

```
           ┌──────────────────────────────────────────────────┐
           │                  TRAINING LOOP                   │
           │                                                  │
           │  1. Forward Pass  ──→  prediction                │
           │        ↓                                         │
           │  2. Compare to target  ──→  loss                 │
           │        ↓                                         │
           │  3. Backward Pass  ──→  gradients                │
           │        ↓                                         │
           │  4. Update Weights  ──→  better next time        │
           │        ↓                                         │
           │  5. Repeat from step 1                           │
           └──────────────────────────────────────────────────┘
```

### Your Snake Training Loop (Annotated)

```python
# train.py
while rounds <= 1000:
    total_rewards = 0
    state = env.reset()          # Get initial state (14 features)
    
    while not env.is_over():
        # 1. PICK ACTION
        action = agent.choose_action(state, epsilon)
        #    (epsilon-greedy: explore vs. exploit)
        
        # 2. TAKE ACTION
        next_state, reward, over = env.step(action)
        total_rewards += reward
        
        # 3. STORE EXPERIENCE
        agent.store(state, action, reward, next_state, over)
        #    (saves to replay buffer: deque of 10,000 memories)
        
        # 4. LEARN FROM BATCH
        agent.train()
        #    (samples 64 random memories, does forward+backward+update)
        
        state = next_state
    
    # 5. SYNC TARGET NETWORK (every 10 episodes)
    if rounds % 10 == 0:
        agent.sync_target()
    
    # 6. DECAY EPSILON (less exploration over time)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    print(f"{rounds}: score-> {env.score} | epsilon-> {epsilon}")
    rounds += 1
```

---

## 10. From Neural Networks to Deep Q-Networks (DQN)

### What Makes DQN Special?

A regular neural network learns from labeled data: `(input → correct_output)`.

A DQN learns from **experience**: `(state → best_action)` — but it doesn't know which action is "correct" in advance. It figures it out through trial and error.

### The Q-Value: What's the Action Worth?

```
Q(s, a) = Expected future reward if you take action 'a' in state 's'
```

```
State:                                 Q-Values:
┌──────────────────┐                  ┌──────────────────┐
│ Snake heading up │                  │ Up:    2.3       │
│ Apple is below   │    ──→ DQN ──→   │ Down:  8.1   ← BEST
│ Free space right │                  │ Left:  4.5       │
└──────────────────┘                  │ Right: 3.2       │
```

The DQN doesn't just predict "what happens next" — it predicts the **total cumulative future reward** for each action.

### The Bellman Equation (The Magic Formula)

This is the core idea behind Q-learning:

```
Q(s, a) = reward + γ × max(Q(next_state, next_action))
           ↑           ↑
    Immediate      Future rewards discounted
    reward         by γ (gamma = 0.95 in your agent)
```

**Intuition:** The value of an action = the immediate reward + (discounted) best future value.

If gamma = 0.95:
- $10 now = $10
- $10 in one step = $9.50 (worth a bit less)
- $10 in ten steps = $5.99 (uncertainty grows)

### The Two-Network Architecture (Your Implementation)

Your agent has TWO identical networks:

```python
self.net         # ONLINE network: learns continuously, picks actions
self.target_net  # TARGET network: stable reference for computing targets
```

#### Why Two Networks?

Without a target network, it's like chasing a moving target:

```
Problem:
Network predicts Q(s, a) = 5
But target uses:        Q(s, a) = reward + γ×Q(next_state, next_action)
                              = 2 + 0.95 × 5.1 = 6.8

The target changes every time the network updates! 
Like a dog chasing its own tail.
```

**Solution:** Freeze the target network and only update it periodically.

```python
# In your agent:
def sync_target(self):
    self.target_net.load_state_dict(self.net.state_dict())
```

Every 10 episodes, you copy the online network's weights to the target network.

### The Replay Buffer: Learning from Past Experiences

```python
self.xp = deque(maxlen=10000)  # Stores up to 10,000 experiences
```

Each experience is a tuple:
```python
(state, action, reward, next_state, game_over)
```

**Why replay past experiences?**
1. **Breaks correlation** — consecutive experiences are very similar (the snake just moves one step). Random sampling breaks this.
2. **Learns from rare events** — crashing into a wall is rare but important. Replay ensures it's seen multiple times.
3. **Data efficiency** — one experience can be used for training multiple times.

### What Your Network Actually Learns

After training, your network doesn't "know" the rules of Snake. It has learned:

1. **Danger avoidance** → going toward walls or yourself → bad Q-value
2. **Apple approach** → getting closer to apple → high Q-value
3. **Efficient paths** → not taking unnecessary risks
4. **Survival** → staying alive is usually better than dying for an apple

---

## 11. Advanced Concepts in Your Snake Agent

Let's decode the specific techniques in your codebase.

### Feature Engineering (snake_env.py: get_state)

Your state vector has 14 features:

```python
# snake_env.py
def get_state(self):
    danger = self.get_danger()       # [straight, left, right] → 3 features
    free_space = self.get_free_space()  # normalized reachable cells → 3 features
    direction = self.snake.get_direction()  # [up, down, left, right] → 4 features
    apple = self.get_apple()         # relative direction to apple → 4 features
    
    return danger + free_space + direction + apple  # total: 14 features
```

**Why 14 features and not the raw pixel grid?**

The original Snake DQN by DeepMind used raw pixels (84×84×4 = 28,224 inputs). Your version uses **hand-crafted features** — a smart engineering decision for a small project:

- **Much faster training** (70k parameters vs millions)
- **Less data needed** (thousands of games vs millions)
- **Better generalization** — features already encode what matters

### Get Danger (snake_env.py: get_danger)

This is a **rule-based perception** system that tells the network what's dangerous:

```python
def get_danger(self):
    danger = [0, 0, 0]  # [straight, right, left]
    
    # Is there a wall straight ahead?
    if (snkHeadX - SIZE < 0 and dir == 'left') or ...:
        danger[straight] = 1
    
    # Is there self-collision to the left?
    if (self.is_self(snkHeadX + SIZE, snkHeadY) and dir == 'up'):
        danger[left] = 1
```

The network doesn't have to figure out what "danger on the left" means — we pre-compute it. This is called **feature engineering** and it dramatically simplifies learning.

### Get Free Space (snake_env.py: get_free_space)

This is an **advanced feature** — a BFS (breadth-first search) that counts reachable cells:

```python
def count_reachable(self, startX, startY):
    # BFS flood fill to count how many cells are reachable
    queue = [(startGridX, startGridY)]
    count = 0
    while queue:
        (x, y) = queue.pop(0)
        if self.is_self(x * SIZE, y * SIZE):
            continue  # Can't go through snake body
        visited.add((x, y))
        count += 1
        queue.append((x+1, y))  # Check all 4 directions
        ...
    return count / max_cells  # Normalize to [0, 1]
```

**Why this is brilliant:** It tells the network not just "is there immediate danger" but "how trapped am I?" A high free_space value means the snake has room to maneuver. A low value means it's boxed in.

### Epsilon-Greedy Exploration (dqn_agent.py: choose_action)

```python
def choose_action(self, state, epsilon):
    randf = random.random()
    if randf < epsilon:
        return random.randint(0, 3)  # EXPLORE: random action
    else:
        q_values = self.net(state)   # EXPLOIT: best known action
        return torch.argmax(q_values).item()
```

```
Early training (ε = 1.0):           Late training (ε ≈ 0.01):
  100% random actions                 1% random actions
  "Let me try everything"            "I know what works, almost always"
  Learning by exploration             Optimizing by exploitation
```

The epsilon decay schedule:
```python
epsilon = 1.0        # Start: all exploration
epsilon_min = 0.01   # End: 1% exploration (just in case)
epsilon_decay = 0.995  # ε = ε × 0.995 each episode

# After 100 episodes: ε ≈ 0.61
# After 500 episodes: ε ≈ 0.08
# After 900 episodes: ε ≈ 0.01 (minimum)
```

### Gradient Clipping (dqn_agent.py: train)

```python
torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
```

**Why:** Sometimes the gradient can be enormous (e.g., when the snake unexpectedly crashes). This is called an **exploding gradient**. Clipping limits each gradient's magnitude to 1.0, preventing wildly destructive updates.

### Discount Factor Gamma (γ = 0.95)

```python
gamma = 0.95
```

Controls how much the network cares about **future vs. immediate rewards**:

| γ | Behavior | Analogy |
|---|----------|---------|
| 0.0 | Only cares about now | Instant gratification |
| 0.5 | Near-sighted | Planning 1-2 steps ahead |
| **0.95** | **Far-sighted** | **Planning ~20 steps ahead** |
| 0.99 | Very far-sighted | Planning ~100 steps ahead |
| 1.0 | Cares infinitely about future | Never converges |

γ = 0.95 means the snake considers roughly the next 20 steps when making decisions — enough to plan efficient paths to the apple.

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **Neuron** | Basic processing unit; computes weighted sum + activation |
| **Weight** | Connection strength between two neurons |
| **Bias** | Offset term that shifts the activation threshold |
| **Layer** | Collection of parallel neurons |
| **Hidden Layer** | Layer between input and output (not directly observed) |
| **Activation Function** | Non-linear function applied to neuron output |
| **ReLU** | `max(0, x)` — most common activation function |
| **Forward Propagation** | Computing output from input through all layers |
| **Backpropagation** | Computing gradients using the chain rule |
| **Loss Function** | Measures prediction error (MSE in your agent) |
| **Gradient Descent** | Algorithm that updates weights to minimize loss |
| **Learning Rate** | Step size for weight updates (0.001 in your agent) |
| **Epoch** | One complete pass through the training data |
| **Batch** | Subset of data used per training step (64 in your agent) |
| **DQN** | Deep Q-Network — neural network that learns Q-values |
| **Q-Value** | Expected cumulative future reward for an action |
| **Bellman Equation** | Q(s,a) = r + γ·max(Q(s',a')) |
| **Replay Buffer** | Memory of past experiences for training (10,000 in your agent) |
| **Target Network** | Frozen copy of the network used for stable targets |
| **Epsilon-Greedy** | Exploration strategy: random with probability ε |
| **Epsilon Decay** | Gradually reducing exploration over time |
| **Gamma (γ)** | Discount factor for future rewards (0.95 in your agent) |
| **Gradient Clipping** | Limiting gradient size to prevent instability |
| **Parameter** | Learnable value in a neural network (weights + biases) |
| **Optimizer** | Algorithm that performs gradient descent (Adam in your agent) |
| **Vanishing Gradient** | Gradients become too small in deep networks, halting learning |
| **Exploding Gradient** | Gradients become too large, destabilizing training |

---

## 13. Further Reading

### Foundational Papers (Linked)
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602) — The original DQN paper
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://nature.com/articles/nature14236) — Nature paper, Nature version of DQN
- [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) — The Bible of deep learning
- [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/) — Excellent free online book

### Interactive Visualizations
- [TensorFlow Playground](https://playground.tensorflow.org/) — Visualize neural networks in your browser
- [3Blue1Brown Neural Network Videos](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Best visual explanation on YouTube
- [CS231n Course Notes](https://cs231n.github.io/) — Stanford's deep learning course

### Key PyTorch Documentation
- [PyTorch nn.Module](https://pytorch.org/docs/stable/nn.html) — Your agent's base class
- [PyTorch optim.Adam](https://pytorch.org/docs/stable/optim.html) — Your optimizer
- [PyTorch MSELoss](https://pytorch.org/docs/stable/nn.html#mseloss) — Your loss function

### Next Steps in Reinforcement Learning
- **Double DQN**: Reduces overestimation bias in Q-learning
- **Dueling DQN**: Separates state-value and action-advantage
- **Prioritized Replay**: Sample important experiences more frequently
- **Rainbow DQN**: Combines all improvements

---

## Quick Reference: Your Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR SNAKE AGENT PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Environment (snake_env.py)                                  │
│      └─ Renders game, computes 14-feature state                 │
│      └─ Gives rewards: +20 for apple, -20 for crash, -1 per step│
│                                                                  │
│   2. Agent (dqn_agent.py)                                       │
│      └─ Neural net: 14 → 256 → 256 → 4 (70,660 parameters)     │
│      └─ Replay buffer: 10,000 memories                          │
│      └─ Target network: sync'd every 10 episodes                │
│      └─ Optimizer: Adam (lr=0.001)                             │
│      └─ Loss: MSELoss                                           │
│                                                                  │
│   3. Training Loop (train.py)                                    │
│      └─ 1000+ episodes                                          │
│      └─ Epsilon: 1.0 → 0.01 (decay ×0.995/episode)             │
│      └─ Batch size: 64, Gamma: 0.95                             │
│      └─ Gradient clipping: max_norm=1.0                         │
│                                                                  │
│   4. Play (play.py)                                              │
│      └─ Loads saved weights, runs 20 episodes with display      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

*Happy learning, and may your snake eat many apples! 🐍🍎*
