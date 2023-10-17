import pandas as pd
from sklearn.model_selection import train_test_split
import torch

import logging

loggerupds = logging.getLogger('update')

# Classes
class CropDataset:
    def __init__(self, data_fname, test_size=0.2, validation_size=0.1):
        self.df = pd.read_csv(data_fname).dropna()
        self.dfX_train, self.dfX_test, self.dfy_train, self.dfy_test = train_test_split(self.df.iloc[:, :-1],
                                                                                        self.df.iloc[:, -1],
                                                                                        test_size=test_size)
        self.dfX_train, self.dfX_validation, self.dfy_train, self.dfy_validation = train_test_split(self.dfX_train,
                                                                                                    self.dfy_train,
                                                                                                    test_size=validation_size)
        print(f'Number of Samples [Total] : {self.df.shape[0]}')
        print(f'Number of Samples [Train] : {self.dfX_train.shape[0]}')
        print(f'Number of Samples [Validation] : {self.dfX_validation.shape[0]}')
        print(f'Number of Samples [Test] : {self.dfX_test.shape[0]}')
        loggerupds.info(f'Number of Samples [Total] : {self.df.shape[0]}')
        loggerupds.info(f'Number of Samples [Train] : {self.dfX_train.shape[0]}')
        loggerupds.info(f'Number of Samples [Validation] : {self.dfX_validation.shape[0]}')
        loggerupds.info(f'Number of Samples [Test] : {self.dfX_test.shape[0]}')

    def get_next_batch(self, use_data='train', batch_size=100):
        if use_data == 'train':
            X = self.dfX_train.copy()
            y = self.dfy_train.copy()
        elif use_data == 'validation':
            X = self.dfX_validation.copy()
            y = self.dfy_validation.copy()
        elif use_data == 'test':
            X = self.dfX_test.copy()
            y = self.dfy_test.copy()
        else:
            raise ValueError(f'use_data can only be one of "train", "validation", "test". Cannot be "{use_data}"')

        data = pd.concat([X, y], axis=1)
        positives = data[data.label == 1].iloc[:, :-1]
        negatives = data[data.label == 0].iloc[:, :-1]
        positives = positives.sample(frac=1)

        pos_batch_size = batch_size // 2
        neg_batch_size = batch_size - pos_batch_size

        start = 0
        num_positive_samples = positives.shape[0]

        while start < num_positive_samples:
            end = start + pos_batch_size
            positive_batch = positives.iloc[start:end]
            # positive_batch['label'] = 1
            positive_batch = positive_batch.assign(label=1)

            if neg_batch_size > len(negatives):
                replace = True
            else:
                replace = False
            negative_batch = negatives.sample(n=neg_batch_size, replace=replace)
            # negative_batch['label'] = 0
            negative_batch = negative_batch.assign(label=0)

            batch = pd.concat([positive_batch, negative_batch], axis=0)
            batch = batch.sample(frac=1)
            batchX = batch.iloc[:, :-1]
            batchy = batch.iloc[:, -1]
            batchX, batchy = torch.tensor(batchX.values).float(), torch.tensor(batchy.values).long()
            batchX = batchX.view(-1, 2, batchX.shape[1]//2)
            # batchX = batchX.view(-1, 2, 15)
            batchy = batchy.view(-1, 1)
            start += batch_size
            yield batchX.transpose(2, 1), batchy