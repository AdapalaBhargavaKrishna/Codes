# The McCulloch-Pitts model is one of the earliest models of a neuron.
# It performs a weighted sum of inputs, applies a threshold, and outputs either 1 or 0.
# For an OR gate,
# Output = 1 if any one or both inputs are 1
# Output = 0 if both inputs are 0

def mcCullochPitts_OR(x1, x2):
    S = x1 + x2

    if S >= 1:
        return 1   # Neuron fires (output 1)
    else:
        return 0   # Neuron does not fire (output 0)

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("OR Function using McCulloch-Pitts Neuron")
for x1, x2 in inputs:
    output = mcCullochPitts_OR(x1, x2)
    print(f"Input: ({x1}, {x2}) -> Output: {output}")
    