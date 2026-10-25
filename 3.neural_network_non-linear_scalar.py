import random

def leaky_relu(x):
    return max(0.01 * x, x)

def leaky_relu_derivative(x):
    return 1 if x > 0 else 0.01

# TRAINING PHASE
SCALE = 100

training_data = []
for _ in range(100):
    x1 = random.randint(1, 100)
    x2 = random.randint(1, 100)
    target = abs(x1 - x2)
    training_data.append((x1/SCALE, x2/SCALE, target/SCALE))

learning_rate = 0.01

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

# neuron 4
w41 = random.uniform(-1, 1)
w42 = random.uniform(-1, 1)
b4 = random.uniform(-1, 1)

# neuron 5
w51 = random.uniform(-1, 1)
w52 = random.uniform(-1, 1)
b5 = random.uniform(-1, 1)

# neuron 6
w61 = random.uniform(-1, 1)
w62 = random.uniform(-1, 1)
b6 = random.uniform(-1, 1)

# neuron 7
w71 = random.uniform(-1, 1)
w72 = random.uniform(-1, 1)
b7 = random.uniform(-1, 1)

# output neuron
w81 = random.uniform(-1, 1)
w82 = random.uniform(-1, 1)
w83 = random.uniform(-1, 1)
w84 = random.uniform(-1, 1)
w85 = random.uniform(-1, 1)
w86 = random.uniform(-1, 1)
w87 = random.uniform(-1, 1)
b8 = random.uniform(-1, 1)

for epoch in range(10000):
    total_loss = 0.0

    for x1, x2, target in training_data:
        # Forward pass
        z1 = (x1 * w11) + (x2 * w12) + b1
        neuron_1 = leaky_relu(z1)
        z2 = (x1 * w21) + (x2 * w22) + b2
        neuron_2 = leaky_relu(z2)
        z3 = (x1 * w31) + (x2 * w32) + b3
        neuron_3 = leaky_relu(z3)
        z4 = (x1 * w41) + (x2 * w42) + b4
        neuron_4 = leaky_relu(z4)
        z5 = (x1 * w51) + (x2 * w52) + b5
        neuron_5 = leaky_relu(z5)
        z6 = (x1 * w61) + (x2 * w62) + b6
        neuron_6 = leaky_relu(z6)
        z7 = (x1 * w71) + (x2 * w72) + b7
        neuron_7 = leaky_relu(z7)

        old_output = (neuron_1 * w81) + (neuron_2 * w82) + (neuron_3 * w83) + (neuron_4 * w84) + (neuron_5 * w85) + (neuron_6 * w86) + (neuron_7 * w87) + b8

        # loss
        error = target - old_output
        old_loss = error ** 2
        total_loss += old_loss

        # Gradient
        gradient_w81 = 2 * (old_output - target) * neuron_1
        gradient_w82 = 2 * (old_output - target) * neuron_2
        gradient_w83 = 2 * (old_output - target) * neuron_3
        gradient_w84 = 2 * (old_output - target) * neuron_4
        gradient_w85 = 2 * (old_output - target) * neuron_5
        gradient_w86 = 2 * (old_output - target) * neuron_6
        gradient_w87 = 2 * (old_output - target) * neuron_7

        gradient_b8 = 2 * (old_output - target)

        gradient_w11 = 2 * (old_output - target) * w81 * leaky_relu_derivative(z1) * x1
        gradient_w12 = 2 * (old_output - target) * w81 * leaky_relu_derivative(z1) * x2
        gradient_b1 = 2 * (old_output - target) * w81 * leaky_relu_derivative(z1)

        gradient_w21 = 2 * (old_output - target) * w82 * leaky_relu_derivative(z2) * x1
        gradient_w22 = 2 * (old_output - target) * w82 * leaky_relu_derivative(z2) * x2
        gradient_b2 = 2 * (old_output - target) * w82 * leaky_relu_derivative(z2)

        gradient_w31 = 2 * (old_output - target) * w83 * leaky_relu_derivative(z3) * x1
        gradient_w32 = 2 * (old_output - target) * w83 * leaky_relu_derivative(z3) * x2
        gradient_b3 = 2 * (old_output - target) * w83 * leaky_relu_derivative(z3)

        gradient_w41 = 2 * (old_output - target) * w84 * leaky_relu_derivative(z4) * x1
        gradient_w42 = 2 * (old_output - target) * w84 * leaky_relu_derivative(z4) * x2
        gradient_b4 = 2 * (old_output - target) * w84 * leaky_relu_derivative(z4)

        gradient_w51 = 2 * (old_output - target) * w85 * leaky_relu_derivative(z5) * x1
        gradient_w52 = 2 * (old_output - target) * w85 * leaky_relu_derivative(z5) * x2
        gradient_b5 = 2 * (old_output - target) * w85 * leaky_relu_derivative(z5)

        gradient_w61 = 2 * (old_output - target) * w86 * leaky_relu_derivative(z6) * x1
        gradient_w62 = 2 * (old_output - target) * w86 * leaky_relu_derivative(z6) * x2
        gradient_b6 = 2 * (old_output - target) * w86 * leaky_relu_derivative(z6)

        gradient_w71 = 2 * (old_output - target) * w87 * leaky_relu_derivative(z7) * x1
        gradient_w72 = 2 * (old_output - target) * w87 * leaky_relu_derivative(z7) * x2
        gradient_b7 = 2 * (old_output - target) * w87 * leaky_relu_derivative(z7)

        # Update output neuron

        w81 = w81 - learning_rate * gradient_w81
        w82 = w82 - learning_rate * gradient_w82
        w83 = w83 - learning_rate * gradient_w83
        w84 = w84 - learning_rate * gradient_w84
        w85 = w85 - learning_rate * gradient_w85
        w86 = w86 - learning_rate * gradient_w86
        w87 = w87 - learning_rate * gradient_w87
        b8 = b8 - learning_rate * gradient_b8

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

        # Update neuron 4

        w41 = w41 - learning_rate * gradient_w41
        w42 = w42 - learning_rate * gradient_w42
        b4 = b4 - learning_rate * gradient_b4

        # Update neuron 5

        w51 = w51 - learning_rate * gradient_w51
        w52 = w52 - learning_rate * gradient_w52
        b5 = b5 - learning_rate * gradient_b5

        # Update neuron 6

        w61 = w61 - learning_rate * gradient_w61
        w62 = w62 - learning_rate * gradient_w62
        b6 = b6 - learning_rate * gradient_b6

        # Update neuron 7

        w71 = w71 - learning_rate * gradient_w71
        w72 = w72 - learning_rate * gradient_w72
        b7 = b7 - learning_rate * gradient_b7

# New prediction
for x1, x2, target in training_data:

    new_z1 = (x1 * w11) + (x2 * w12) + b1
    new_neuron_1 = leaky_relu(new_z1)
    new_z2 = (x1 * w21) + (x2 * w22) + b2
    new_neuron_2 = leaky_relu(new_z2)
    new_z3 = (x1 * w31) + (x2 * w32) + b3
    new_neuron_3 = leaky_relu(new_z3)
    new_z4 = (x1 * w41) + (x2 * w42) + b4
    new_neuron_4 = leaky_relu(new_z4)
    new_z5 = (x1 * w51) + (x2 * w52) + b5
    new_neuron_5 = leaky_relu(new_z5)
    new_z6 = (x1 * w61) + (x2 * w62) + b6
    new_neuron_6 = leaky_relu(new_z6)
    new_z7 = (x1 * w71) + (x2 * w72) + b7
    new_neuron_7 = leaky_relu(new_z7)

    output = ((new_neuron_1 * w81) + (new_neuron_2 * w82) + (new_neuron_3 * w83) + (new_neuron_4 * w84) + (new_neuron_5 * w85) + (new_neuron_6 * w86) + (new_neuron_7 * w87) + b8)
    target = target
    loss = (target - output) ** 2

    print("Input:", x1 * SCALE, x2 * SCALE, "Target:", target * SCALE, "New Output:", output * SCALE, "Loss:", loss)

# TESTING PHASE
x1 = 13 / SCALE
x2 = 7 / SCALE
target = 6 / SCALE

new_z1= (x1 * w11) + (x2 * w12) + b1
new_neuron_1 = leaky_relu(new_z1)
new_z2 = (x1 * w21) + (x2 * w22) + b2
new_neuron_2 = leaky_relu(new_z2)
new_z3 = (x1 * w31) + (x2 * w32) + b3
new_neuron_3 = leaky_relu(new_z3)
new_z4 = (x1 * w41) + (x2 * w42) + b4
new_neuron_4 = leaky_relu(new_z4)
new_z5 = (x1 * w51) + (x2 * w52) + b5
new_neuron_5 = leaky_relu(new_z5)
new_z6 = (x1 * w61) + (x2 * w62) + b6
new_neuron_6 = leaky_relu(new_z6)
new_z7 = (x1 * w71) + (x2 * w72) + b7
new_neuron_7 = leaky_relu(new_z7)

output = ((new_neuron_1 * w81) + (new_neuron_2 * w82) + (new_neuron_3 * w83) + (new_neuron_4 * w84) + (new_neuron_5 * w85) + (new_neuron_6 * w86) + (new_neuron_7 * w87) + b8)
target = target
loss = (target - output) ** 2

print("Output:", output * SCALE,"Loss:", loss)
