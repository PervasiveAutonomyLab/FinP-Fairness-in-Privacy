import numpy as np  # Importing NumPy for numerical and array operations.
import torch  # Importing PyTorch for tensor operations and building machine learning models.
import pandas as pd  # Importing pandas for data manipulation and analysis.
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting datasets.
from torch.utils.data import TensorDataset, DataLoader  # Importing TensorDataset for creating PyTorch datasets and DataLoader for batching.
import logging  # Importing logging for debug and trace information.
import random



# Configuring logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

mean_std_dict = {
    'learning_state': (0.3, 0.1),  # Mean and std for learning_state
    'drowsiness_state': (0.4, 0.15),  # Mean and std for drowsiness_state
    'dizziness_sickness_state': (0.5, 0.2)  # Mean and std for dizziness_sickness_state
}

def augment_row(row, n_samples=60):
    augmented_rows = []
    for _ in range(n_samples):
        augmented_row = row.copy()

        # Generate values for each feature using normal distribution
        for feature in ['learning_state', 'drowsiness_state', 'dizziness_sickness_state']:
            mean, std = mean_std_dict[feature]
            random_value = np.random.normal(mean, std)
            # If the original value is 0, keep generating until we get a value below 0.6
            if row[feature] == 0:
                while random_value >= 0.6:
                    random_value = np.random.normal(mean, std)
            else:
                while random_value < 0.6:
                    random_value = np.random.normal(mean, std)
            augmented_row[feature] = float(round(random_value, 3))

        augmented_rows.append(augmented_row)

    return augmented_rows

# Apply augmentation to each row and combine into a single DataFrame



def load_custom_data():
    df = pd.read_csv('edu_dataset/action_log.csv')
    augmented_data = []
    for _, row in df.iterrows():
        augmented_data.extend(augment_row(row))
    augmented_df = pd.DataFrame(augmented_data)
    logging.info("Data loaded into DataFrame")
    return augmented_df

# Main function to process HAR data and prepare it for training
def process_edu_data(num_subjects: int = 3, random_activity_sample = True):
    set_seed(30)  # Set seed for reproducibility
    train_data = load_custom_data()  # Load data
    train_data.reset_index(drop=True, inplace=True)
    sampled_number = 30


    # Maps subject IDs to sequential indices
    unique_subjects = train_data['subject'].unique()  # Gets unique subject IDs.

    # Based on tayler request Randomly select a subset of subjects if num_subjects is specified
    if num_subjects < len(unique_subjects):
        selected_subjects = np.random.choice(unique_subjects, num_subjects, replace=False)
        logging.info(f'Selected {num_subjects} subjects out of {len(unique_subjects)} available.')

        # Filter train and test data to include only the selected subjects
        train_data = train_data[train_data['subject'].isin(selected_subjects)].reset_index(drop=True)
        subject_mapping = {subject: idx for idx, subject in enumerate(sorted(selected_subjects))}
    else:
        subject_mapping = {subject: idx for idx, subject in enumerate(unique_subjects)}  # Maps each unique subject to an index.



    # Applies the mapping to train and test data
    train_data['subject'] = train_data['subject'].map(subject_mapping)
    logging.info('Mapping subjects to indices complete.')

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

    if random_activity_sample:
        logging.info('Randomly sampling records for each activity per subject.')
        sampled_data = []

        # Define baseline sample size for each subject to distribute across activities
        baseline_sample_size = 100  # Adjust based on the total data or desired skew
        min_records_per_activity = 5  # Minimum records per activity to ensure representation

        for subject in train_data['subject'].unique():
            subject_data = train_data[train_data['subject'] == subject]

            # activity_proportions = np.random.dirichlet(alpha=alpha_values)

            # for activity in train_data['y'].unique():
            for idx, action in enumerate(train_data['action'].unique()):
                actions_data = subject_data[subject_data['action'] == action]
                #
                # activity_data = subject_data[subject_data['action'] == activity]
                #
                # # Calculate sample size based on the Dirichlet proportion and baseline sample size
                # sample_size = max(min_records_per_activity, int(baseline_sample_size * activity_proportions[idx]))
                #
                # # Ensure we do not sample more records than available
                # sample_size = min(sample_size, len(activity_data))
                #
                # # Randomly sample the specified number of records
                # selected_records = activity_data.sample(n=sample_size, random_state=42) if len(
                #     activity_data) > 1 else activity_data


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
                if len(actions_data) > 1:
                    selected_records = actions_data.sample(n=random.randint(1, len(actions_data)),
                                                            random_state=20)
                else:
                    selected_records = actions_data

                sampled_data.append(selected_records)

            # Combine all sampled records into a single DataFrame
        train_data = pd.concat(sampled_data).reset_index(drop=True)
        logging.info('Random sampling of records per activity complete.')

    # Prepare features and labels
    X = train_data[['learning_state', 'drowsiness_state', 'dizziness_sickness_state']].values
    y = train_data['action'].values -  train_data['action'].min() # Adjust labels to start from 0

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X).float()
    y_train_tensor = torch.from_numpy(y).long()


    logging.info('Data conversion to NumPy arrays complete.')


    logging.info('Conversion to PyTorch tensors complete.')

    # Creates TensorDataset objects for train and test data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # Combines training features and labels into a PyTorch dataset.

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
        train_indices, test_indices = train_test_split(subject_indices, test_size=0.3, random_state=42)
        sampled_indices = np.random.choice(subject_indices, size=min(sampled_number, len(subject_indices)), replace=False)
        sampled_subject_record[subject] = sampled_indices.tolist()  # Stores sampled indices.

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

    # Create a combined test dataset from accumulated indices
    X_combined_test = X_train_tensor[combined_test_indices]
    y_combined_test = y_train_tensor[combined_test_indices]
    test_dataset = TensorDataset(X_combined_test, y_combined_test)

    logging.info('Combined test dataset created from accumulated indices.')

    # Final return and log
    logging.info('Processing of EDU Data data complete.')
    return train_dataset, test_dataset, subject_train_indices, sampled_subject_record, subject_test_indices, label_test_datasets,subject_labels_count