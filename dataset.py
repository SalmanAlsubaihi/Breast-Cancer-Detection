from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision
from tqdm import tqdm
from pathlib import Path, PureWindowsPath
import numpy as np

class ToFloat():
    def __call__(self, tensor):
        return tensor.type(torch.float32)

class HorizontalFlip():
    def __call__(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

class ApplyWindow():
    def __init__(self, window_type):
        self.window_type = window_type
    def __call__(self, img, img_info):
        window_center = img_info[f'{self.window_type}_Center']
        window_width = img_info[f'{self.window_type}_Width']
        window_min, window_max = window_center - window_width/2, window_center + window_width/2
        return img.clip(min=window_min, max=window_max)

class ApplyWindowNormalize():
    def __init__(self, window_type):
        self.window_type = window_type
    def __call__(self, img, img_info):
        window_center = img_info[f'{self.window_type}_Center']
        window_width = img_info[f'{self.window_type}_Width']
        window_min, window_max = window_center - window_width/2, window_center + window_width/2
        img = img.clip(min=window_min, max=window_max)
        img = img - img.min()
        img = img/img.max()
        return img

class Normalize():
    def __call__(self, img):
        img = img - img.min()
        img = img/img.max()
        return img

def get_avg_size(csv_file_path, root_dir):
    dataset = BcDataset(csv_file_path, root_dir, transformations=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
    all_sizes = []
    img_keys = ['L_CC', 'R_CC', 'L_MLO', 'R_MLO']
    for dict_ in dataloader:
        for k in img_keys:
            all_sizes.append(list(dict_[k].shape)[2:4])
    d1, d2 = zip(*all_sizes)
    avg_size = [int(np.array(d1).mean()), int(np.array(d2).mean())]
    return avg_size

def get_mean_and_std(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(dataloader):
        data = torch.cat([data['L_CC'], data['L_MLO'], data['R_CC'], data['R_MLO']])
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std


class BcDatasetLocal():
    def __init__(self, csv_file_path, root_dir, classification_task, transformations=None):
        self.df = pd.read_csv(csv_file_path)
        self.list_patients = self.df.P_ID.unique().tolist()
        self.root_dir = root_dir
        self.transformations = transformations
        self.label_mapping = local_label_mappings[classification_task]
        self.num_classes = len(set(self.label_mapping.values()))
        self.all_views = [('L', 'CC'), ('L', 'MLO'), ('R', 'CC'), ('R', 'MLO')]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.list_patients[idx]
        patient_dict = self.get_patient_info(patient_id)
        
        label = self.label_mapping[patient_dict['L_CC']['Overall_score']]
        label = torch.tensor(int(label), dtype=torch.float32).long()
        
        patient_dict = self.apply_transformations(patient_dict)
        
        for view in patient_dict:
            patient_dict[view] = patient_dict[view]['image']
        patient_dict.update({'label' : label})
                
        return patient_dict

    def get_patient_info(self, patient_id):
        patient_df = self.df[self.df.P_ID == patient_id]
        patient_dict = {}
        for view in self.all_views:
            patient_dict.update({
                f'{view[0]}_{view[1]}' :
                patient_df[patient_df[['IL', 'VP']].apply(tuple, axis = 1).isin([view])].sample().to_dict('records')[0]
            })
        for view in patient_dict.keys():
            patient_dict[view].update({
                'image' : Image.open(self.root_dir+patient_dict[view]['IMAGE_PATH'])
            })
        return patient_dict
    
    def apply_transformations(self, patient_dict):
        for trans in self.transformations:
            if isinstance(trans, HorizontalFlip):
                for view in patient_dict.keys():  # TODO: use this for all if conditions
                    if view.startswith('L'):
                        patient_dict[view]['image'] = trans(patient_dict[view]['image'])
            elif isinstance(trans, ApplyWindow) or isinstance(trans, ApplyWindowNormalize):
                for view in patient_dict.keys():
                    patient_dict[view]['image'] = trans(patient_dict[view]['image'], patient_dict[view])
            else:
                for view in patient_dict.keys():
                    patient_dict[view]['image'] = trans(patient_dict[view]['image'])
        return patient_dict
    
    def __len__(self):
        return len(self.list_patients)



class BcDatasetMiniDdsm(BcDatasetBase):
    def __init__(self, csv_file_path, root_dir, classification_task, transformations=None, horizontal_flip=True):
        self.df = pd.read_csv(csv_file_path)
        file_names = self.df['fileName'].values
        patients = [f.split('.')[0] for f in file_names]
        self.df['P_ID'] = patients
        self.list_patients = self.df.P_ID.unique().tolist()
        self.root_dir = root_dir
        self.transformations = transformations
        self.horizontal_flip = horizontal_flip
        self.label_mapping = mini_ddsm_label_mappings[classification_task]
        self.num_classes = len(set(self.label_mapping.values()))
    def get_patient_info(self, patient_id):
        patient_info = self.df[self.df.P_ID == patient_id][['fullPath', 'Side', 'View', 'Status']].values.tolist()        
        patient_info = [[PureWindowsPath(path).as_posix(), side[0], view, status]
                        for path, side, view, status in patient_info]
        patient_info = [[path, f'{side}_{view}', status]
                        for path, side, view, status in patient_info]
        label = self.label_mapping[patient_info[0][2]]
        return patient_info, label


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient_id = self.list_patients[idx]
        patient_info, label = self.get_patient_info(patient_id)   
        patient_dict = self.get_four_view_images(patient_info)
        
        if self.horizontal_flip:
            for view in patient_dict.keys():
                if view.startswith('L'):
                    patient_dict[view] = patient_dict[view].transpose(Image.FLIP_LEFT_RIGHT)

        if self.transformations:
            for view in patient_dict.keys():
                patient_dict[view] = self.transformations(patient_dict[view])

#         patient_dict.update({'label': torch.LongTensor([int(label)])})
        patient_dict.update({'label': label})
        return patient_dict

    def __len__(self):
        return len(self.list_patients)

    def get_four_view_images(self, patient_info):
        p_dict = {}
        p_dict.update({'L_CC': self.root_dir + random.choice([path for path, v, _ in patient_info if v == 'L_CC'])})
        p_dict.update({'L_MLO': self.root_dir + random.choice([path for path, v, _ in patient_info if v == 'L_MLO'])})
        p_dict.update({'R_CC': self.root_dir + random.choice([path for path, v, _ in patient_info if v == 'R_CC'])})
        p_dict.update({'R_MLO': self.root_dir + random.choice([path for path, v, _ in patient_info if v == 'R_MLO'])})
        for view in p_dict.keys():
            p_dict[view] = Image.open(p_dict[view])
        return p_dict



dataset_class = {
    'mini_ddsm':BcDatasetMiniDdsm,
    'local_data':BcDatasetLocal,
    'local_data_all_labels':BcDatasetLocal,
    'local_data_alnaeem':BcDatasetLocal,
    'local_data_all_labels_meta':BcDatasetLocal,
}

dataset_paths = {
    'local_data_alnaeem':{'train_csv_file_path':'new_dataframes/Batch_2_FOR_PRESENTATION_cropped_labeled_windows_Dr. Abdulrahman Alnaeem_tumer_train.csv',
                  'val_csv_file_path':'new_dataframes/Batch_2_FOR_PRESENTATION_cropped_labeled_windows_Dr. Abdulrahman Alnaeem_tumer_val.csv',
                  'root_dir':'../breast_cancer_data/'},

    'local_data_all_labels':{'train_csv_file_path':'new_dataframes/Batch_3_FOR_PRESENTATION_cropped_labeled_tumer_train.csv',
                  'val_csv_file_path':'new_dataframes/Batch_3_FOR_PRESENTATION_cropped_labeled_tumer_val.csv',
                  'root_dir':'../breast_cancer_data/'},

    'local_data_all_labels_meta':{'train_csv_file_path':'new_dataframes/Batch_2_FOR_PRESENTATION_cropped_labeled_tumer_meta_2_train.csv',
                  'val_csv_file_path':'new_dataframes/Batch_2_FOR_PRESENTATION_cropped_labeled_tumer_meta_2_val.csv',
                  'root_dir':'../breast_cancer_data/'},

    'mini_ddsm':{'train_csv_file_path':'DataWMask_train_short.csv',
                  'val_csv_file_path':'DataWMask_val.csv',
                  'root_dir':'MINI-DDSM-Complete-JPEG-8/'}
}

local_label_mappings = {
    'tumer': {'1 - negative':0, '2 - benign':1}
}

mini_ddsm_label_mappings = {
    'tumer': {'Normal':0, 'Benign':1, 'Cancer':1},  ## TODO: include 'Cancer'
    'normal_benign_cancer' : {'Normal':0, 'Benign':1, 'Cancer':2}
}