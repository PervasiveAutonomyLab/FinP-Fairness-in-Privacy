import os
import pickle

class Tracker:
    def __init__(self, args,class_num):
        self.args = args
        self.class_num = class_num
        self.client_accuracy = {i: [-1] * self.args.epochs for i in range(self.args.num_users)}
        self.sia_accuracy =  {i: [-1] * self.args.epochs for i in range(self.args.num_users)}
        self.model_accuracy_train = [-1 for _ in range(self.args.epochs)]
        self.model_accuracy_test = [-1 for _ in range(self.args.epochs)]
        self.sia_accuracy_avg = [-1 for _ in range(self.args.epochs)]
        self.class_accuracy = {i: [-1] * self.args.epochs for i in range(self.class_num)}
        self.time = [-1 for _ in range(self.args.epochs)]

        self.confidence_all = [ [] for i in range(self.args.epochs)]

    def add_model_accuracy_test(self, iteration, accuracy):
        self.model_accuracy_test[iteration] = accuracy

    def add_model_accuracy_train(self, iteration, accuracy):
        self.model_accuracy_train[iteration] = accuracy

    def add_sia_accuracy(self, iteration, client_index, accuracy):
        self.sia_accuracy[client_index][iteration] = accuracy

    def add_sia_accuracy_avg(self, iteration, accuracy):
        self.sia_accuracy_avg[iteration] = accuracy

    def add_client_accuracy(self, iteration, client_index, accuracy):
        self.client_accuracy[client_index][iteration] = accuracy

    def add_class_accuracy(self, iteration, class_index, accuracy):
        self.class_accuracy[class_index][iteration] = accuracy

    def add_confidence_all(self, iteration, confidence):
        self.confidence_all[iteration] = confidence
    def add_time(self, iteration, time):
        self.time[iteration] = time

    def save_results(self, folder, filename='saved_results_1.pkl'):
        file_path = os.path.join(folder, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb+') as f:
            pickle.dump([self.sia_accuracy, self.model_accuracy_train, self.model_accuracy_test,
                          self.confidence_all, self.sia_accuracy_avg, self.time], f)
        
        print(f'Results saved at {file_path}')