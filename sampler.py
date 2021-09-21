from torch.utils.data.sampler import BatchSampler
import numpy as np
import csv

class BalancedBatchSampler(BatchSampler):
    def __init__(self, file_path, n_classes, n_samples):
        self.labels_list = []
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                self.labels_list.append(int(row[1]))

        self.labels = np.array(self.labels_list)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set} #get indices for each class from the csv file

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
            temp_arr = self.label_to_indices[l].copy()
            self.label_to_indices[l] = np.append(self.label_to_indices[l], temp_arr)

        self.used_label_indices_count = {label: 0 for label in self.labels_set} #acts as pointer for tracking used indices
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples #samples per class
        self.len_dataset = len(self.labels_list)
        self.batch_size = self.n_samples * self.n_classes
        self.all_classes = [i for i in range(0, self.n_classes)]
        np.random.shuffle(self.all_classes)
        self.idx_class = 0

    def __iter__(self):
        self.count = 0  #to track if dataset has been explored
        while self.count + self.batch_size < self.len_dataset:
            np.random.shuffle(self.all_classes)
            indices = [] #collects the batch
            for class_ in self.all_classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples #move pointer for a particular class

                # reset pointer if all indices visited for that class
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    indices_set = list(set(self.label_to_indices[class_].copy()))
                    np.random.shuffle(indices_set)
                    self.label_to_indices[class_] = np.append(self.label_to_indices[class_], indices_set)
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.len_dataset // self.batch_size

def get_sampler(config_data):
    sampler_src_train = BalancedBatchSampler(config_data['path'], config_data['num_categories'], config_data['n_samples'])
    return sampler_src_train

