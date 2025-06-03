import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader
import torch
from .options import args_parser

args = args_parser()


def testSubsets(test_dataset, desired_distribution):
    rng = np.random.default_rng(seed=args.manualseed)

    x_test = []
    y_test = []
    for image, label in test_dataset:
        x_test.append(image.numpy())  # Convert the image tensor to a numpy array and store in list
        y_test.append(label)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Number of classes
    num_classes = len(desired_distribution[0])
    num_users = len(desired_distribution)

    subsets = []

    # print('y_test', y_test)

    # Number of samples in the desired test subset
    alpha = 0.1
    total_test_samples = alpha * int(len(y_test))
    # print('tts', total_test_samples)
    for user in range(num_users):
        # Initialize lists to store the selected indices
        subset_indices = []

        distribution = np.array(normalize(np.array(desired_distribution[user]).reshape(1, -1), norm='l1')).flatten()
        print('distribution', user, desired_distribution[user], distribution)

        # Loop through each class and sample according to the desired distribution
        for class_label in range(num_classes):
            # Get all indices for the current class label in the test set
            class_indices = np.where(y_test == class_label)[0]
            # print('class_indices', class_label, class_indices)
            # Calculate the number of samples needed for the test subset for this class
            num_class_samples = int(total_test_samples * distribution[class_label])
            ## print('num', class_indices, len(class_indices), num_class_samples)

            # Randomly sample the required number of indices for this class
            selected_indices = rng.choice(class_indices, size=min(num_class_samples, len(class_indices)), replace=False)

            # Add these indices to the final list
            subset_indices.extend(selected_indices)

        # Convert to array to use for indexing
        subset_indices = np.array(subset_indices)
        # print('subset_indices', subset_indices)

        # print('len of subset', len(x_test[subset_indices]), len(y_test[subset_indices]))

        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_test[subset_indices]).float(), torch.from_numpy(y_test[subset_indices]).long()))
        subsets.append(test_dataset.dataset)
        # print(subsets[0])
    return subsets
        # Create the final test subset
        # X_test_subset = X_test[subset_indices]
        # y_test_subset = y_test[subset_indices]


def per_class_subsets(test_dataset, desired_distribution):
    rng = np.random.default_rng(seed=args.manualseed)

    x_test = []
    y_test = []
    for image, label in test_dataset:
        x_test.append(image.numpy())  # Convert the image tensor to a numpy array and store in list
        y_test.append(label)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    num_classes = len(desired_distribution[0])
    num_users = len(desired_distribution)

    alpha = 0.5
    total_test_samples = alpha * int(len(y_test))
    subsets = []
    for class_label in range(num_classes):
        # Get all indices for the current class label in the test set
        class_indices = np.where(y_test == class_label)[0]
        # Calculate the number of samples needed for the test subset for this class
        num_class_samples = int(total_test_samples/num_classes)

        # Randomly sample the required number of indices for this class
        selected_indices = rng.choice(class_indices, size=min(num_class_samples, len(class_indices)),
                                            replace=False)

        # Convert to array to use for indexing
        subset_indices = np.array(selected_indices)

        # print('len of subset', len(x_test[subset_indices]), len(y_test[subset_indices]))

        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_test[subset_indices]).float(),
                                                torch.from_numpy(y_test[subset_indices]).long()))
        subsets.append(test_dataset.dataset)
        # print('subset_indices',subset_indices)
    return subsets






