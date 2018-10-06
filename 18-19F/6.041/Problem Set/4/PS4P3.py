import random

H = 1
T = 0
p = 0.5
n = 100000

flips = random.choices([H, T], [p, 1 - p], k=n)

rewards = 0

for i in range(1, n):
    if flips[i - 1] == H and flips[i] == T:
        rewards += 1

print('Expected Mean: ' + str((n - 1) * p * (1 - p)))
print('Actual Mean: ' + str(rewards))

