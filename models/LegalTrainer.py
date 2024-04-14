import os, sys
import time, json, math
import numpy as np
import torch
import torch.optim as optim
import logging
import random
import matplotlib.pyplot as plt
import pickle
from models.AttnCut import AttnCut
from models.Choopy import Choopy
from models.LeCut import LeCut
from models.MileCut import MileCut
from models.BiCut import BiCut
from models.MtCut import MtCut
from utils import losses
from utils.metrics import Metric, Metric_for_Loss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from datetime import datetime
from models.trainer_utils import seed_it, get_oracle_pos

level_relations = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}
def _get_logger(filename, level='info'):
    log = logging.getLogger(filename)
    log.setLevel(level_relations.get(level))
    fmt = logging.Formatter('%(asctime)s: %(message)s')
    file_handler = logging.FileHandler(filename, 'a', encoding='utf-8')
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)
    return log

log_path = f'logs/trainer_{time.strftime("%Y%m%d-%H-%M")}.log'
logger = _get_logger(log_path, 'info')

class LegalTrainer:
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        logger.info(f"cuda: {self.cuda}")
        self.dataset_seed = 42
        self.train_seed = 42

        # Truncation model params
        self.model_name = 'MileCut'
        self.loss_name = 'MileCut'
        self.epoch = 100
        self.batch_size = 20
        self.lr = 3e-5
        self.dropout = 0.2
        self.weight_decay = float('0.005')
        self.input_size = 6
        self.coefficient = 0.6
        self.seq_len = 100

        self.save_path = 'models/output'
        self.truncation_model_save_path = self.save_path + '/{}.pkl'.format(self.model_name + time.strftime("%Y%m%d-%H-%M"))
        self.model_persist = True
        self.best_test_f1 = -float('inf')
        self.best_test_dcg = -float('inf')
        self.best_test_oie = -float('inf')
        self.f1_record = []
        self.dcg_record = []
        self.oie_record = []

    def prepare_lecard_dataset(self, path, set_seed=True):  # LeCaRD
        with open(path, 'r', encoding='utf-8') as f:
            lecard_dataset = json.load(f)
            f.close()
        if set_seed:
            seed_it(self.dataset_seed)
        all_index = list(lecard_dataset.keys())
        train_index = np.sort(random.sample(all_index, 87))
        test_index = np.sort(list(set(all_index).difference(set(train_index))))
        self.dataset_train = [lecard_dataset[i] for i in train_index]
        self.dataset_test = [lecard_dataset[i] for i in test_index]
        logging.info(f'prepare dataset, train size: {len(train_index)}, test size: {len(test_index)}')

    def prepare_coliee_dataset(self, path, set_seed=True):
        if set_seed:
            seed_it(self.seed)
        train_index = list(range(718))
        test_index = list(range(718, 898))
        dataset_train, dataset_test = [], []
        for i in tqdm(train_index, desc='prepare_coliee_dataset train'):
            with open(os.path.join(path, f'{i}.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset_train.append(data)
                f.close()
        for i in tqdm(test_index, desc='prepare_coliee_dataset test'):
            with open(os.path.join(path, f'{i}.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset_test.append(data)
                f.close()
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        logging.info(f'prepare_coliee_dataset, train size: {len(train_index)}, test size: {len(test_index)}')

    def prepare_civil_dataset_mini(self, path, set_seed=True):
        if set_seed:
            seed_it(self.dataset_seed)
        train_index = list(range(8))
        test_index = list(range(8, 10))
        dataset_train, dataset_test = [], []
        for i in tqdm(train_index, desc='prepare_civil_dataset train'):
            with open(os.path.join(path, f'{i}.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset_train.append(data)
                f.close()
        for i in tqdm(test_index, desc='prepare_civil_dataset test'):
            with open(os.path.join(path, f'{i}.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset_test.append(data)
                f.close()
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        logging.info(f'prepare_civil_dataset, train size: {len(train_index)}, test size: {len(test_index)}')

    def train_epoch(self, epoch):
        epoch_loss, epoch_f1, epoch_dcg = 0, 0, 0
        epoch_oie = 0
        step, num_itr = 0, len(self.train_loader)
        for X_train, y_train in tqdm(self.train_loader, desc='Training for epoch_{}'.format(epoch)):
            self.model.train()
            self.optimizer.zero_grad()
            if self.cuda: X_train, y_train = X_train.cuda(), y_train.cuda()
            if self.model_name == 'MileCut':
                truncation_output, view_1_output, view_2_output, view_3_output = self.model(X_train)
                loss = self.criterion(truncation_output, view_1_output, view_2_output, view_3_output, y_train)
                output = truncation_output
            elif self.model_name == 'Choopy' or self.model_name == 'AttnCut' or self.model_name == 'LeCut' or self.model_name == 'BiCut' or self.model_name == 'MtCut':
                output = self.model(X_train)
                loss = self.criterion(output, y_train)

            loss.backward()
            self.optimizer.step()
            if self.model_name == 'BiCut':
                predictions = np.argmax(output.detach().cpu().numpy(), axis=2)
                k_s = []
                for results in predictions:
                    if np.sum(results) == self.seq_len: k_s.append(self.seq_len)
                    else: k_s.append(np.argmin(results)+1)
            elif self.model_name == 'MtCut':
                predictions = output[-1].detach().cpu().squeeze().numpy()
                k_s = np.argmax(predictions, axis=1) + 1
            else:
                predictions = output.detach().cpu().squeeze().numpy()
                if output.shape[0] == 1:
                    predictions = predictions.reshape(1, predictions.shape[0])
                k_s = np.argmax(predictions, axis=1) + 1
            y_train_np = y_train.data.cpu().numpy()
            f1 = Metric.f1(y_train_np, k_s)
            dcg = Metric.dcg(y_train_np, k_s)
            oie = Metric.oie(y_train_np, k_s)

            epoch_loss += loss.item()
            epoch_f1 += f1
            epoch_dcg += dcg
            epoch_oie += oie
            step += 1

        train_loss, train_f1, train_dcg = epoch_loss / step, epoch_f1 / step, epoch_dcg / step
        train_oie = epoch_oie / step
        self.train_loss.append(train_loss)
        if epoch % 5 == 0:
            logger.info('epoch: {}, Train: loss = {} | f1 = {:.6f} | dcg = {:.6f} | oie = {:.6f}'.format(epoch, train_loss, train_f1, train_dcg, train_oie))

    def test(self, epoch):
        epoch_loss, epoch_f1, epoch_dcg = 0, 0, 0
        epoch_oie = 0
        step = 0
        oracle_pos = []
        pre_pos = []
        for X_test, y_test in tqdm(self.test_loader, desc='Test after epoch_{}'.format(epoch)):
            self.model.eval()
            with torch.no_grad():
                if self.cuda: X_test, y_test = X_test.cuda(), y_test.cuda()
                if self.model_name == 'MileCut':
                    truncation_output, view_1_output, view_2_output, view_3_output = self.model(X_test)
                    loss = self.criterion(truncation_output, view_1_output, view_2_output, view_3_output, y_test)
                    output = truncation_output
                elif self.model_name == 'Choopy' or self.model_name == 'AttnCut' or self.model_name == 'LeCut' or self.model_name == 'BiCut' or self.model_name == 'MtCut':
                    output = self.model(X_test)
                    loss = self.criterion(output, y_test)
    
                if self.model_name == 'BiCut':
                    predictions = np.argmax(output.detach().cpu().numpy(), axis=2)
                    k_s = []
                    for results in predictions:
                        if np.sum(results) == self.seq_len: k_s.append(self.seq_len)
                        else: k_s.append(np.argmin(results) + 1)
                elif self.model_name == 'MtCut':
                    predictions = output[-1].detach().cpu().squeeze().numpy()
                    k_s = np.argmax(predictions, axis=1) + 1
                else:
                    predictions = output.detach().cpu().squeeze().numpy()
                    if output.shape[0] == 1:
                        predictions = predictions.reshape(1, predictions.shape[0])
                    k_s = np.argmax(predictions, axis=1) + 1

                y_test_np = y_test.data.cpu().numpy()
                f1 = Metric.f1(y_test_np, k_s)
                dcg = Metric.dcg(y_test_np, k_s)
                oie = Metric.oie(y_test_np, k_s)

                pre_pos = pre_pos + list(k_s)
                o_s, _, _ = get_oracle_pos(y_test_np, self.seq_len)
                oracle_pos = oracle_pos + list(o_s)

                epoch_loss += loss.item() * y_test_np.shape[0]
                epoch_f1 += f1 * y_test_np.shape[0]
                epoch_dcg += dcg * y_test_np.shape[0]
                epoch_oie += oie * y_test_np.shape[0]
                step += 1

        data_len = len(self.test_loader.dataset)
        test_loss, test_f1, test_dcg = epoch_loss / data_len, epoch_f1 / data_len, epoch_dcg / data_len
        test_oie = epoch_oie / data_len
        self.test_loss.append(test_loss)
        if epoch % 5 == 0 or epoch == self.epoch-1:
            logger.info('epoch: {}, Test: loss = {} | f1 = {:.6f} | dcg = {:.6f} | oie = {:.6f}'.format(epoch, test_loss, test_f1, test_dcg, test_oie))

        self.f1_record.append(test_f1)
        self.dcg_record.append(test_dcg)
        self.oie_record.append(test_oie)

        if test_f1 > self.best_test_f1:
            self.best_test_f1 = test_f1
            logger.info('epoch: {} | best_test_f1 = {:.6f} | dcg = {:.6f} | oie = {:.6f}'.format(epoch, test_f1, test_dcg, test_oie))
            if self.model_persist: self.save_model()

        if test_dcg > self.best_test_dcg: self.best_test_dcg = test_dcg
        if test_oie > self.best_test_oie: self.best_test_oie = test_oie

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(self.model.state_dict(), self.truncation_model_save_path)
        logger.info('The best model has beed updated and saved in {}\n'.format(self.truncation_model_save_path))

    def load_model(self):
        self.model.load_state_dict(torch.load(self.truncation_model_save_path))
        logger.info('The best model has beed loaded from {}\n'.format(self.truncation_model_save_path))

    def run(self, reset_params=False, lr=3e-5, dropout=0.2, weight_decay=float('0.005'), epoch=50, loss_name='MileCut', model_name='MileCut', coefficient=0.6, input_size=6, set_train_seed=True, train_seed=42, view_input_size=3, label_input_size=3, seq_len=100):
        if reset_params:
            self.epoch = epoch
            self.lr = lr
            self.dropout = dropout
            self.weight_decay = weight_decay
            self.model_name = model_name
            self.loss_name = loss_name
            self.coefficient = coefficient
            self.input_size = input_size
            self.view_input_size = view_input_size
            self.label_input_size = label_input_size
            self.seq_len = seq_len
        if set_train_seed:
            self.train_seed = train_seed
            seed_it(self.train_seed)
        logging.info('\nTrain the {} model: \n'.format(self.model_name))
        self.truncation_model_save_path = self.save_path + '/{}.pkl'.format(self.model_name + time.strftime("%Y%m%d-%H-%M"))
        logger.info(
            f'epoch: {self.epoch}, lr:{self.lr}, reset_params: {reset_params}, dropout:{self.dropout}, weight_decay:{self.weight_decay}, model_name: {self.model_name}, loss function: {self.loss_name}, loss coefficient: {self.coefficient}, input_size: {self.input_size}, batch_size: {self.batch_size}, set_train_seed={set_train_seed}, train_seed={train_seed}, view_input_size={self.view_input_size}, label_input_size: {label_input_size}')

        if self.model_name == 'MileCut':
            self.model = MileCut(input_size=self.input_size, dropout=self.dropout, view_input_size=self.view_input_size, label_input_size=self.label_input_size, seq_len=self.seq_len)

        elif self.model_name == 'AttnCut':
            self.model = AttnCut(input_size=self.input_size)
        elif self.model_name == 'Choopy':
            self.model = Choopy(seq_len=self.seq_len)
        elif self.model_name == 'LeCut':
            self.model = LeCut(input_size=self.input_size, seq_len=self.seq_len)
        elif self.model_name == 'BiCut':
            self.model = BiCut(input_size=self.input_size)
        elif self.model_name == 'MtCut':
            self.model = MtCut(input_size=self.input_size, seq_len=self.seq_len)

        if self.loss_name == 'MileCut':
            self.criterion = losses.MileCutLoss(metric='f1', coefficient=self.coefficient)
        elif self.loss_name == 'AttnCut':
            self.criterion = losses.AttnCutLoss(metric='f1')
        elif self.loss_name == 'Choopy':
            self.criterion = losses.ChoopyLoss(metric='f1')
        elif self.loss_name == 'LeCut':
            self.criterion = losses.LeCutLoss(metric='f1')
        elif self.loss_name == 'BiCut':
            self.criterion = losses.BiCutLoss(metric='f1')
        elif self.loss_name == 'MtCut':
            self.criterion = losses.MtCutLoss(metric='f1')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.best_test_f1 = -float('inf')
        self.best_test_dcg = -float('inf')
        self.best_test_oie = -float('inf')
        self.f1_record = []
        self.dcg_record = []
        self.oie_record = []
        self.train_loss = []
        self.test_loss = []
        for epoch in range(self.epoch):
            self.train_epoch(epoch)
            self.test(epoch)
        best5_f1 = sum(sorted(self.f1_record, reverse=True)[:5]) / 5
        best5_dcg = sum(sorted(self.dcg_record, reverse=True)[:5]) / 5
        best5_oie = sum(sorted(self.oie_record, reverse=True)[:5]) / 5
        logger.info('the best metric of this model: f1: {} | dcg: {} | oie: {}'.format(self.best_test_f1, self.best_test_dcg, self.best_test_oie))
        logger.info('the best-5 metric of this model: f1: {} | dcg: {} | oie: {}'.format(best5_f1, best5_dcg, best5_oie))

        plt.figure()
        plt.plot(self.train_loss, label="train loss")
        plt.show()

        plt.figure()
        plt.plot(self.test_loss, label="test loss")
        plt.show()




