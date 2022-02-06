import os
import numpy as np
import pandas as pd
from datetime import datetime
import random
import copy
import time
import argparse

#Visualization imports
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 18})
# plt.style.use('ggplot')
sns.set_theme(color_codes=True)
print(matplotlib.get_backend())

#Preprocessing imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#NN imports
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

class Trainer(object):
    def __init__(self, lr, epochs, visualize, train, test, seed_number, schedule, device):

        self.LR = lr
        self.epochs = epochs
        self.viz = visualize
        self.train_set = train
        self.test_set = test
        self.seed = seed_number
        self.schedule = schedule
        self.device = device
        self.path = os.getcwd()

    def impute_and_plot(self):
        #Check missing values
        if self.viz:
            plt.figure(figsize = (25,11))
            sns.heatmap(self.train_set.isna().values, xticklabels=self.train_set.columns)
            plt.title("Missing values in training Data", size=20)
        categorical_features = ['State_Factor', 'building_class', 'facility_type']
        numerical_features = self.train_set.select_dtypes('number').columns
        #Fill missing values == Data Impute 
        #code copied from https://www.kaggle.com/shrutisaxena/wids2022-starter-code
        missing_columns = [col for col in self.train_set.columns if self.train_set[col].isnull().any()]
        missingvalues_count = self.train_set.isna().sum()
        missingValues_df = pd.DataFrame(missingvalues_count.rename('Null Values Count')).loc[missingvalues_count.ne(0)]
        # missingValues_df.style.background_gradient(cmap="Pastel1")

        self.train_set['year_built'] = self.train_set['year_built'].replace(np.nan, 2022)
        self.test_set['year_built'] = self.test_set['year_built'].replace(np.nan, 2022)
        null_col=['energy_star_rating','direction_max_wind_speed','direction_peak_wind_speed','max_wind_speed','days_with_fog']
        imputer = SimpleImputer()
        imputer.fit(self.train_set[null_col])
        data_transformed = imputer.transform(self.train_set[null_col])
        self.train_set[null_col] = pd.DataFrame(data_transformed)
        test_data_transformed = imputer.transform(self.test_set[null_col])
        self.test_set[null_col] = pd.DataFrame(test_data_transformed)

        #Encode the string features
        le = LabelEncoder()
        for col in categorical_features:
            self.train_set[col] = le.fit_transform(self.train_set[col])
            self.test_set[col] = le.fit_transform(self.test_set[col])
            
        #Visualize the dataset
        if self.viz:
            self.train_set.describe()
            plt.figure(figsize = (25,11))
            sns.heatmap(self.train_set.isna().values, xticklabels=self.train_set.columns)
            plt.title("Missing values in training Data", size=20)

        return self.train_set, self.test_set

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        print('Done SEEDing: {}'.format(self.seed))

    def create_tensor_dataset(self, tr, te):

        Y_target = tr["site_eui"].to_numpy()
        train_f = tr.drop(["site_eui","id"],axis=1)
        test_f_id = te['id']
        test_f = te.drop(["id"],axis=1)

        scaler = StandardScaler()
        train_f = scaler.fit_transform(train_f)
        test_f = scaler.transform(test_f)
        print(train_f, train_f.shape, Y_target, Y_target.shape)
        # print(test_f, test_f.shape, test_f_id, test_f_id.shape)

        #Split data into TRAIN and TEST 80:20
        x_train, x_test, y_train, y_test = train_test_split(train_f, Y_target, test_size = 0.2, random_state = self.seed)
        # print(x_train, x_test.shape, y_train, y_test, y_test.shape)
        x_train_tensor = torch.Tensor(x_train)
        x_test_tensor = torch.Tensor(x_test)
        y_train_tensor = torch.Tensor(y_train)
        y_test_tensor = torch.Tensor(y_test)
        #Evaluation set for submission
        eval_target = torch.Tensor(test_f_id.to_numpy())
        y_eval = torch.Tensor(test_f)

        #Create Tensor datasets out of numpy dataset 
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        eval_dataset = TensorDataset(y_eval, eval_target)

        #Create the dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=2)

        return train_dataloader, test_dataloader, eval_loader

    def train_one_epoch(self, epoch_index, tb_writer, train_dataloader):
        running_loss = 0.
        last_loss = 0.
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            outputs = self.model(inputs.to(device))
            # Compute the loss and its gradients
            loss = self.loss_function(outputs.squeeze(1), labels.to(device))
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            if i % 800 == 799:
                last_loss = running_loss / 800 # loss per batch
    #             print('Mphka kai vrhka to last loss:{}'.format(last_loss))
    #             print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(train_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        
        return last_loss

    def evaluate_for_submission(self):
        paths = []
        predictions_eui = []
        #Evaluation for submission
        for dirname, _, filenames in os.walk(self.path + '/saves'):
            for filename in filenames:
                paths.append(os.path.join(dirname, filename))
        model_dir = paths[-1]
        print(model_dir)
        self.model.load_state_dict(torch.load(model_dir))
        self.model.eval()

        for i, vdata in enumerate(self.eval_loader):
            vinputs, _ = vdata
            voutputs = self.model(vinputs.to(device))
            predictions_eui.append(voutputs.detach().cpu().numpy())

        out = np.concatenate(predictions_eui).ravel()

        # print(len(out), out)
        submission = pd.read_csv('widsdatathon2022/sample_solution.csv')
        submission['site_eui'] = out
        submission.head()

        submission.to_csv('submission.csv', index=False)

    def train_multiple_epochs(self):
        #Setup the model
        n_features = 62
        hidden_size = 300
        self.model = SimpleMLP(n_features, hidden_size)
        self.model.to(self.device)
        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.optimizer = optim.Adam(params, lr=self.LR)
        # and a learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        # loss_function = nn.MSELoss()
        # loss = F.mse_loss()
        self.loss_function = nn.L1Loss()

        train_set_edited, test_set_edited = self.impute_and_plot()
        self.train_dataloader, self.test_dataloader, self.eval_loader = self.create_tensor_dataset(train_set_edited, test_set_edited)
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/wids_trainer_{}'.format(timestamp))
        save_path = os.path.join("saves")
        os.makedirs(save_path, exist_ok=True)
        epoch_number = 0
        EPOCHS = self.epochs
        loss_list = []
        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))
            ts = time.time()
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer, self.train_dataloader)
            # We don't need gradients on to do reporting
            self.model.train(False)
            self.model.eval()
            
            te = time.time()
            running_vloss = 0.0
            for i, vdata in enumerate(self.test_dataloader):
                vinputs, vlabels = vdata
                voutputs = self.model(vinputs.to(device))
                vloss = self.loss_function(voutputs.squeeze(1), vlabels.to(device))
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('Train Loss {} Validation Loss {}'.format(avg_loss, avg_vloss))
        #     print('I was trained on {} for {} and evaluated for {} seconds'.format(device, round(train_time, 3), round(time.time() - te, 3)))
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()
            # Track best performance, and save the model's state
            avg_mse = avg_vloss.detach().cpu().numpy() / 1.0
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = save_path + '/model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)
            
            loss_list.append([avg_loss, avg_mse])
            if self.schedule:
                self.lr_scheduler.step()
            epoch_number += 1
        train_time = round((time.time() - ts)/60, 2)
        mean_losses = pd.DataFrame(loss_list, columns=['Average_Training_Loss', 'Average_Validation_Loss'])
        #Plot the train and validation losses.
        # sns.relplot(x='epoch', y='Average_Validation_Loss',  kind='line', data=mean_losses)
        # sns.relplot(x='epoch', y='Average_Training_Loss',  kind='line', data=mean_losses)
        print('Trained for {} minutes'.format(train_time))
        if self.viz:            
            plt.figure() 
            mean_losses.plot()
            plt.legend(loc='best')


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleMLP, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        obs = self.net(x)
        return obs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visual", required=False, default=True, help="Visualize the features")
    parser.add_argument("-e", "--epochs", required=False, default=400, help="Epochs to train on")
    parser.add_argument("-l", "--lr", required=False, default=1e-4, help="Optimizer's Learning Rate")
    parser.add_argument("-s", "--seed", required=False, default=2022, help="Set seed for reproducibility")
    parser.add_argument("-sc", "--schedule", required=False, default=False, help="Use learning rate scheduler")
    args = parser.parse_args()
    #Read the dataset
    train_set = pd.read_csv("widsdatathon2022/train.csv")
    test_set = pd.read_csv("widsdatathon2022/test.csv")
    print("Number of train samples are", train_set.shape)
    print("Number of test samples are", test_set.shape)
    #Select the device to train on
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Training on {}".format(device))
    #Execute the training procedure
    trainer = Trainer(args.lr, args.epochs, args.visual, train_set, test_set, args.seed, args.schedule, device)
    trainer.seed_everything()
    trainer.train_multiple_epochs()
    trainer.evaluate_for_submission()