import random

# TRAINING PHASE
SCALE = 100

training_data = []
for _ in range(100):
    x1 = random.randint(1, 100)
    x2 = random.randint(1, 100)
    target = (x1 + x2)
    training_data.append((x1/SCALE, x2/SCALE, target/SCALE))

learning_rate = 0.001

# neuron 1
w11 = random.uniform(-1, 1)
w12 = random.uniform(-1, 1)
b1 = random.uniform(-1, 1)

# neuron 2
w21 = random.uniform(-1, 1)
w22 = random.uniform(-1, 1)
b2 = random.uniform(-1, 1)

# neuron 3
w31 = random.uniform(-1, 1)
w32 = random.uniform(-1, 1)
b3 = random.uniform(-1, 1)

# output neuron
w41 = random.uniform(-1, 1)
w42 = random.uniform(-1, 1)
w43 = random.uniform(-1, 1)
b4 = random.uniform(-1, 1)

for epoch in range(10000):
    total_loss = 0.0

    for x1, x2, target in training_data:
        # Forward pass
        neuron_1 = (x1 * w11) + (x2 * w12) + b1
        neuron_2= (x1 * w21) + (x2 * w22) + b2
        neuron_3 = (x1 * w31) + (x2 * w32) + b3
    
        old_output = (neuron_1 * w41) + (neuron_2 * w42) + (neuron_3 * w43) + b4

        # loss
        error = target - old_output
        old_loss = error ** 2
        total_loss += old_loss

        # Gradient
        gradient_w41 = 2 * (old_output - target) * neuron_1
        gradient_w42 = 2 * (old_output - target) * neuron_2
        gradient_w43 = 2 * (old_output - target) * neuron_3
        gradient_b4 = 2 * (old_output - target)

        gradient_w11 = 2 * (old_output - target) * w41 * x1
        gradient_w12 = 2 * (old_output - target) * w41 * x2
        gradient_b1 = 2 * (old_output - target) * w41

        gradient_w21 = 2 * (old_output - target) * w42 * x1
        gradient_w22 = 2 * (old_output - target) * w42 * x2
        gradient_b2 = 2 * (old_output - target) * w42

        gradient_w31 = 2 * (old_output - target) * w43 * x1
        gradient_w32 = 2 * (old_output - target) * w43 * x2
        gradient_b3 = 2 * (old_output - target) * w43

        # Update output neuron

        w41 = w41 - learning_rate * gradient_w41
        w42 = w42 - learning_rate * gradient_w42
        w43 = w43 - learning_rate * gradient_w43
        b4 = b4 - learning_rate * gradient_b4

        # Updated neuron 1

        w11 = w11 - learning_rate * gradient_w11
        w12 = w12 - learning_rate * gradient_w12
        b1 = b1 - learning_rate * gradient_b1

        # Update neuron 2

        w21 = w21 - learning_rate * gradient_w21
        w22 = w22 - learning_rate * gradient_w22
        b2 = b2 - learning_rate * gradient_b2

        # Update neuron 3

        w31 = w31 - learning_rate * gradient_w31
        w32 = w32 - learning_rate * gradient_w32
        b3 = b3 - learning_rate * gradient_b3

# New prediction
for x1, x2, target in training_data:

    new_neuron_1 = (x1 * w11) + (x2 * w12) + b1
    new_neuron_2 = (x1 * w21) + (x2 * w22) + b2
    new_neuron_3 = (x1 * w31) + (x2 * w32) + b3

    output = ((new_neuron_1 * w41) + (new_neuron_2 * w42) + (new_neuron_3 * w43) + b4)
    loss = (target - output) ** 2

    print("Input:", x1 * SCALE, x2 * SCALE, "Target:", target * SCALE, "New Output:", output * SCALE, "Loss:", loss)

# TESTING PHASE
x1 = 343/SCALE
x2 = 7/SCALE
target = 350/SCALE

new_neuron_1 = (x1 * w11) + (x2 * w12) + b1
new_neuron_2 = (x1 * w21) + (x2 * w22) + b2
new_neuron_3 = (x1 * w31) + (x2 * w32) + b3


output = ((new_neuron_1 * w41) + (new_neuron_2 * w42) + (new_neuron_3 * w43) + b4)
target = target
loss = (target - output) ** 2

print("Prediction:", output * SCALE,"Loss:", loss)
