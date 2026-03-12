from load_dataset import load_dataset

X, y, classes = load_dataset()

print("Total images:", len(X))
print("Total labels:", len(y))
print("Total classes:", len(classes))