import matplotlib.pyplot as plt

train_acc = [0.8567,0.9127,0.9287,0.9418,0.9520,
             0.9604,0.9674,0.9712,0.9748,0.9776]

val_acc = [0.9075,0.9071,0.9254,0.9343,0.9304,
           0.9359,0.9309,0.9358,0.9359,0.9360]

epochs = range(1,11)

plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig("outputs/accuracy_graph.png")

plt.show()