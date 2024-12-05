from torchvision.datasets import CIFAR10

# mnist_dataset_train = MNIST("./", train=True, download=True)
# mnist_dataset_test = MNIST("./", train=False, download=True)

# mnist_dataset_train = CIFAR10("./", train=True, download=True)
mnist_dataset_test = CIFAR10("./", train=False, download=True)

print(mnist_dataset_test.classes)
# import pathlib
#


# d = pathlib.Path("CIFAR10")
# d.mkdir(exist_ok=True)
# d_train = d / "train" / "images"
# d_train.mkdir(exist_ok=True, parents=True)
# d_val = d / "validation" / "images"
# d_val.mkdir(exist_ok=True, parents=True)
# labels = {}
# for i in range(len(mnist_dataset_train)):
#     image, label = mnist_dataset_train[i]
#     image.save(d_train / f"{i}.jpg")
#     labels[f"{i}.jpg"] = label
# with open(d / "train" / "targets.pkl", "wb") as f:
#     import pickle
#     pickle.dump(labels, f)
#
# labels = {}
# for i in range(len(mnist_dataset_test)):
#     image, label = mnist_dataset_test[i]
#     image.save(d_val / f"{i}.jpg")
#     labels[f"{i}.jpg"] = label
# with open(d / "validation" / "targets.pkl", "wb") as f:
#     import pickle
#     pickle.dump(labels, f)
