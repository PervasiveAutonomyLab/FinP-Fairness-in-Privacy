import numpy as np  # Importing NumPy for numerical and array operations.
import torch  # Importing PyTorch for tensor operations and building machine learning models.
import pandas as pd  # Importing pandas for data manipulation and analysis.
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting datasets.
from torch.utils.data import TensorDataset, DataLoader  # Importing TensorDataset for creating PyTorch datasets and DataLoader for batching.
import logging  # Importing logging for debug and trace information.
import random
from .options import args_parser
args = args_parser()


# Configuring logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of sensor signal types used in the dataset
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

def set_seed(seed=args.manualseed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function to read a CSV file with space-separated values and no headers
def _read_csv(filename):
    logging.info(f'Reading CSV file: {filename}')
    return pd.read_csv(filename, sep='\s+', header=None)  # Reads a file and returns a DataFrame.

# Function to load label data for a given subset (train/test)
def load_y(subset):
    filename = f'{subset}/y_{subset}.txt'  # Constructs the filename for the labels.
    logging.info(f'Loading labels from: {filename}')
    y = _read_csv(filename)[0]  # Reads the first column as label data.
    return y.to_numpy()  # Converts the DataFrame column to a NumPy array.

# Function to load sensor signal data for a given subset (train/test)
def load_signals(subset):
    logging.info(f'Loading signal data for subset: {subset}')
    signals_data = []  # Initializes an empty list to store signal data.
    for signal in SIGNALS:  # Iterates over each signal type.
        filename = f'{subset}/Inertial Signals/{signal}_{subset}.txt'  # Constructs the filename for the signal.
        logging.info(f'Reading signal data from: {filename}')
        signals_data.append(_read_csv(filename).to_numpy())  # Reads the signal data and adds it to the list.
    logging.info(f'Successfully loaded signal data for subset: {subset}')
    return np.transpose(signals_data, (1, 2, 0))  # Transposes the array to shape (samples, timesteps, features).

# Function to load the entire HAR dataset for both train and test sets
def load_har_data():
    logging.info('Loading HAR dataset...')
    X_train = load_signals('train')  # Loads training signal data.
    X_test = load_signals('test')  # Loads testing signal data.

    y_train = load_y('train')  # Loads training labels.
    y_test = load_y('test')  # Loads testing labels.
    subject_train = _read_csv('train/subject_train.txt').iloc[:, 0]  # Reads training subject IDs.
    subject_test = _read_csv('test/subject_test.txt').iloc[:, 0]  # Reads testing subject IDs.

    # Creates DataFrames for train and test data
    train_data = pd.DataFrame({'subject': subject_train, 'X': list(X_train), 'y': y_train})
    test_data = pd.DataFrame({'subject': subject_test, 'X': list(X_test), 'y': y_test})

    logging.info('Successfully loaded HAR dataset.')
    return train_data, test_data  # Returns the train and test DataFrames.

# Main function to process HAR data and prepare it for training
def process_har_data(combine_train_test: bool = False,balance_dataset: bool = False, num_subjects: int = 30,
                     random_activity_sample = False,alpha_values: list = None):
    set_seed(args.manualseed)  # Set seed at the start of the function for reproducibility
    logging.info('Processing HAR data...')
    train_data, test_data = load_har_data()  # Loads the train and test data.

    # Maps subject IDs to sequential indices
    unique_subjects = pd.concat([train_data['subject'], test_data['subject']]).unique()  # Gets unique subject IDs.

    # Based on tayler request Randomly select a subset of subjects if num_subjects is specified
    if num_subjects < len(unique_subjects):
        selected_subjects = np.random.choice(unique_subjects, num_subjects, replace=False)
        logging.info(f'Selected {num_subjects} subjects out of {len(unique_subjects)} available.')

        # Filter train and test data to include only the selected subjects
        train_data = train_data[train_data['subject'].isin(selected_subjects)].reset_index(drop=True)
        test_data = test_data[test_data['subject'].isin(selected_subjects)].reset_index(drop=True)
        subject_mapping = {subject: idx for idx, subject in enumerate(sorted(selected_subjects))}
    else:
        subject_mapping = {subject: idx for idx, subject in enumerate(unique_subjects)}  # Maps each unique subject to an index.

    # Applies the mapping to train and test data
    train_data['subject'] = train_data['subject'].map(subject_mapping)
    test_data['subject'] = test_data['subject'].map(subject_mapping)

    logging.info('Mapping subjects to indices complete.')

    # Combine train and test data if the flag is set
    if combine_train_test:
        logging.info('Combining training and testing data into one dataset.')
        combined_data = pd.concat([train_data, test_data]).reset_index(drop=True)
        train_data = combined_data  # Use combined data for training

    if balance_dataset:
        logging.info('Balancing dataset by ensuring all subjects have the same number of records for each activity.')

        # Calculate the minimum number of records for each activity across all subjects
        min_records_per_activity = {}
        for activity in train_data['y'].unique():
            activity_counts = train_data[train_data['y'] == activity].groupby('subject').size()
            min_records_per_activity[activity] = activity_counts.min()

        # Balance the dataset for each subject and activity
        balanced_train_data = []

        for subject in train_data['subject'].unique():
            subject_data = train_data[train_data['subject'] == subject]

            for activity in train_data['y'].unique():
                activity_data = subject_data[subject_data['y'] == activity]

                if len(activity_data) > min_records_per_activity[activity]:
                    # Randomly sample records to match the minimum count
                    activity_data = activity_data.sample(n=min_records_per_activity[activity], random_state=42)

                balanced_train_data.append(activity_data)

        # Combine the balanced data into a single DataFrame
        train_data = pd.concat(balanced_train_data).reset_index(drop=True)
        logging.info('Dataset balanced.')

    activity_sampling_probs = {
            1: 0.6,  # Activity 1 gets 30% of the available records
            2: 0.9,  # Activity 2 gets 40%
            3: 0.3,  # Activity 3 gets 50%
            4: 0.8,  # Activity 4 gets 60%
            5: 0.7,  # Activity 5 gets 70%
            6: 0.4,  # Activity 6 gets 80%
        }
    mean_records_per_activity = {
        1: 20,  # Mean for activity 1
        2: 15,  # Mean for activity 2
        3: 30,  # Mean for activity 3
        4: 15,  # Mean for activity 4
        5: 40,  # Mean for activity 5
        6: 50,  # Mean for activity 6
    }

    min_records_per_activity = 20  # Ensure each activity has at least 20 records

        # New logic for random activity sampling
    if random_activity_sample:
        logging.info('Randomly sampling records for each activity per subject.')
        sampled_data = []

        # Define baseline sample size for each subject to distribute across activities
        baseline_sample_size = 100  # Adjust based on the total data or desired skew
        min_records_per_activity = 5  # Minimum records per activity to ensure representation

        for subject in train_data['subject'].unique():
            subject_data = train_data[train_data['subject'] == subject]

            activity_proportions = np.random.dirichlet(alpha=alpha_values)

            # for activity in train_data['y'].unique():
            for idx, activity in enumerate(train_data['y'].unique()):
                activity_data = subject_data[subject_data['y'] == activity]

                activity_data = subject_data[subject_data['y'] == activity]

                # Calculate sample size based on the Dirichlet proportion and baseline sample size
                sample_size = max(min_records_per_activity, int(baseline_sample_size * activity_proportions[idx]))

                # Ensure we do not sample more records than available
                sample_size = min(sample_size, len(activity_data))

                # Randomly sample the specified number of records
                selected_records = activity_data.sample(n=sample_size, random_state=42) if len(
                    activity_data) > 1 else activity_data


                # # Set the sampling probability for the activity
                # sampling_prob = activity_sampling_probs.get(activity, 0.3)
                #
                # # Calculate the sample size: retain a minimum number of records or use the sampling probability
                # sample_size = max(min_records_per_activity, int(len(activity_data) * sampling_prob))
                #
                # # Ensure we do not sample more records than available
                # sample_size = min(sample_size, len(activity_data))
                #
                # # Randomly sample the specified number of records
                # selected_records = activity_data.sample(n=sample_size, random_state=42) if len(
                #     activity_data) > 1 else activity_data

                ########################
                # Randomly select a subset of records for each activity if there are multiple records
                # if len(activity_data) > 1:
                #     selected_records = activity_data.sample(n=random.randint(1, len(activity_data)),
                #                                             random_state=20)
                # else:
                #     selected_records = activity_data

                sampled_data.append(selected_records)

            # Combine all sampled records into a single DataFrame
        train_data = pd.concat(sampled_data).reset_index(drop=True)
        logging.info('Random sampling of records per activity complete.')
    # Converts all subjects' data into single arrays for the DataLoader
    X_train = np.stack(train_data['X'].values)  # Stacks training data into a 3D NumPy array.
    y_train = train_data['y'].values - 1  # Adjusts training labels to start from 0.
    X_test = np.stack(test_data['X'].values)  # Stacks testing data into a 3D NumPy array.
    y_test = test_data['y'].values - 1  # Adjusts testing labels to start from 0.

    logging.info('Data conversion to NumPy arrays complete.')

    # Converts the NumPy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float().permute(0, 2, 1)  # Converts and permutes training data.
    y_train_tensor = torch.from_numpy(y_train).long()  # Converts training labels to a long tensor.
    X_test_tensor = torch.from_numpy(X_test).float().permute(0, 2, 1)  # Converts and permutes testing data.
    y_test_tensor = torch.from_numpy(y_test).long()  # Converts testing labels to a long tensor.

    logging.info('Conversion to PyTorch tensors complete.')

    # Creates TensorDataset objects for train and test data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # Combines training features and labels into a PyTorch dataset.
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # Combines testing features and labels into a PyTorch dataset.

    logging.info('TensorDatasets created for train and test data.')

    # Initializes dictionaries for storing various subsets of data
    subject_test_indices = {}  # Dictionary to store test data for each subject.
    subject_train_indices = {}  # Dictionary to store training indices for each subject.
    sampled_subject_record = {}  # Dictionary to store random sample records for each subject.
    subject_labels_count = {}  # Dictionary to store label counts for each subject.

    label_test_datasets = {}  # Dictionary to store test data for each label.
    label_data = {}  # Temporary dictionary for accumulating test data by label.
    num_classes = y_train_tensor.max().item()  # Determines the number of classes.

    combined_test_indices = []  # List to accumulate test indices if combine_train_test is enabled


    # Split each subject's data into 80% train and 20% test
    for subject, group_data in train_data.groupby('subject'):
        logging.info(f'Processing data for subject: {subject}')
        subject_indices = group_data.index.tolist()  # Gets indices for the current subject.

        y_subject = y_train_tensor[subject_indices]  # Gets labels for the current subject.

        # Count occurrences of each label for this subject
        label_counts = []
        for label in range(0, num_classes + 1):  # Iterates over possible labels.
            label_counts.append((y_subject == label).sum().item())  # Counts occurrences of each label.

        subject_labels_count[subject] = label_counts  # Stores label counts.

        # Split indices into training and testing subsets
        train_indices, test_indices = train_test_split(subject_indices, test_size=0.2, random_state=42)
        sampled_indices = np.random.choice(subject_indices, size=min(100, len(subject_indices)), replace=False)
        sampled_subject_record[subject] = sampled_indices.tolist()  # Stores sampled indices.

        if combine_train_test:
            combined_test_indices.extend(test_indices)  # Accumulate test indices if flag is enabled

        # Extracts data for train and test portions
        X_subject_train = X_train_tensor[train_indices]
        y_subject_train = y_train_tensor[train_indices]
        X_subject_test = X_train_tensor[test_indices]
        y_subject_test = y_train_tensor[test_indices]

        subject_train_indices[subject] = train_indices  # Stores training indices.
        subject_test_indices[subject] = TensorDataset(X_subject_test, y_subject_test)  # Creates a TensorDataset for test data.

        logging.info(f'Data processed for subject: {subject}')

        # Group data by label for test dataset
        for label in torch.unique(y_subject_test):
            label = label.item()
            label_indices = (y_subject_test == label).nonzero(as_tuple=True)[0]

            if label not in label_data:
                label_data[label] = {'X': [], 'y': []}

            label_data[label]['X'].append(X_subject_test[label_indices])
            label_data[label]['y'].append(y_subject_test[label_indices])

        # Convert accumulated data for each label to TensorDatasets
        for label, data in label_data.items():
            X_label = torch.cat(data['X'], dim=0)
            y_label = torch.cat(data['y'], dim=0)
            label_test_datasets[label] = TensorDataset(X_label, y_label)

    logging.info('Label-specific TensorDatasets created.')

    if combine_train_test:
        # Create a combined test dataset from accumulated indices
        X_combined_test = X_train_tensor[combined_test_indices]
        y_combined_test = y_train_tensor[combined_test_indices]
        test_dataset = TensorDataset(X_combined_test, y_combined_test)

        logging.info('Combined test dataset created from accumulated indices.')

    # Final return and log
    logging.info('Processing of HAR data complete.')
    return train_dataset, test_dataset, subject_train_indices, sampled_subject_record, subject_test_indices, label_test_datasets,subject_labels_count