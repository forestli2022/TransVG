import matplotlib.pyplot as plt
import json

base_path = 'outputs/refcoco_'
file_postfixes = ['v0_t0', 'v20_t20', 'v20_t20_drop', 'v20_t20_drop_vl1']  # Adjust as needed

MAX_EPOCH = 20
epochs = [i for i in range(MAX_EPOCH)]

plt.figure(figsize=(10, 6))
for file_postfix in file_postfixes:
    with open(base_path + file_postfix + "/log.txt", 'r') as file:
        cnt = 0
        accuracy = []
        for line in file:
            if cnt >= MAX_EPOCH: break
            # Parse each line as JSON
            try:
                data = json.loads(line.strip())
                accuracy.append(data.get("validation_accu", 0.0))  # Default to 0.0 if 'validation_accu' key is missing
                cnt += 1
            except json.JSONDecodeError:
                # Handle the case where a line is not valid JSON
                print(f"Skipping invalid JSON line: {line}")

        print(len(accuracy), len(epochs))
        plt.plot(epochs, accuracy, marker='o', linestyle='-', label=f'{file_postfix}')

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Epoch')
plt.legend()
plt.grid(True)

plt.savefig("validation_accuracy_plot.png")
plt.close()
