def mcCullochPitts_AND(x1, x2):
    S = x1 + x2   # Sum of inputs

    if S >= 2:    # Threshold = 2
        return 1  # Neuron fires (output 1)
    else:
        return 0  # Neuron does not fire (output 0)

# All possible input combinations
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("AND Function using McCulloch-Pitts Neuron")
for x1, x2 in inputs:
    output = mcCullochPitts_AND(x1, x2)
    print(f"Input: ({x1}, {x2}) -> Output: {output}")
