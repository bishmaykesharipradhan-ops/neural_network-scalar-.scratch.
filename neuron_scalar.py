import random

# TRAINING PHASE
SCALE = 100
learning_rate = 0.001

training_data = []
for _ in range(100):
    x1 = random.randint(1, 100)
    x2 = random.randint(1, 100)
    target = x1 + x2
    training_data.append((x1/SCALE, x2/SCALE, target/SCALE))

w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)

bias = random.uniform(-1, 1)

for epoch in range(10000):
    total_loss = 0.0
    for x1, x2, target in training_data:

        output = (x1 * w1) + (x2 * w2) + bias
        error = target - output
        loss = error ** 2

        total_loss += loss

        w1 = w1 + learning_rate * error * x1
        w2 = w2 + learning_rate * error * x2
        bias = bias + learning_rate * error

for x1, x2, target in training_data:
        output = ((x1 * w1) + (x2 * w2) + bias)
        loss = target - output
        print("Input:", x1 * SCALE, x2 * SCALE,"Target:", target * SCALE,"Prediction:", output * SCALE, "Loss:", loss)
# TESTING PHASE

x1 = 343/SCALE
x2 = 7/SCALE
target = 350/SCALE

output = ((x1 * w1) + (x2 * w2) + bias)
error = target - output
loss = error ** 2
print("Prediction:", output * SCALE,"Loss:", loss)