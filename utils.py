import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from scipy import ndimage

def calc_accuracy_and_loss(net, loss_function, val_dataloader, train_mode):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc_accuracy = train_mode == 'supervised'
    net.eval()
    all_labels = []
    all_pred = []
    all_loss = []
    for batch in val_dataloader:
        with torch.no_grad():
            for k in batch.keys():
                batch[k] = batch[k].to(device)#.double()
            out = net(batch)
            label = batch['label'].long()
            loss = loss_function(out, label)
            all_loss.append(loss.detach().cpu())
            if calc_accuracy:
                pred = out.argmax(dim=1)
                all_labels.append(label.detach().cpu())
                all_pred.append(pred.detach().cpu())
    mean_loss = torch.tensor(all_loss).mean()
    if not calc_accuracy:
        return mean_loss, None
    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)
    accuracy = (all_labels == all_pred).sum()/len(all_labels)
    return mean_loss, accuracy


class ModelSaver():
    def __init__(self, hparams, save_folder, exp_start_time):
        self.exp_start_time = exp_start_time
        self.save_folder = save_folder
        self.best_val_accuracy = 0
        self.best_val_loss = 10**10
        self.train_mode = hparams['train_mode']
        self.backbone_net = hparams['backbone_net']
        self.model = hparams['model']

    def _save(self, net, optimizer, best_in, epoch):
        if self.train_mode == 'supervised':
            checkpoint = {
                'state_dict' : net.state_dict(),
                'model' : self.model,
                'epoch' : epoch,
                'optimizer': optimizer.state_dict(),
                'train_mode' : self.train_mode
            }
        elif self.train_mode == 'self_supervised':
            checkpoint = {
                'state_dict' : net.feature_extractor.state_dict(),
                'backbone_net' : self.backbone_net,
                'epoch' : epoch,
                'optimizer': optimizer.state_dict(),
                'train_mode' : self.train_mode
            }
        else:
            raise NotImplementedError
        save_path = os.path.join(self.save_folder, f'{best_in}_' + self.exp_start_time)
        torch.save(checkpoint, save_path)
    def save_best_model(self, net, optimizer, val_loss, val_accuracy, epoch):
        if val_loss < self.best_val_loss:
            self._save(net, optimizer, 'best_loss', epoch)
        if val_accuracy > self.best_val_accuracy:
            self._save(net, optimizer, 'best_acc', epoch)
        self._save(net, optimizer, 'latest', epoch)



class Loger():
    def __init__(self, hparams):
        self.exp_start_time = '_'.join(time.asctime().split(' ')[1:])
        exp_name = '--'.join([f'{k}={v}' for k,v in hparams.items()])
        self.log_dir = f'log/vio_lut_pre-processing'
        self.writer = SummaryWriter(log_dir=self.log_dir)  ## choose name
        self.all_loss, self.all_pred, self.all_label = [], [], []
        self.train_mode = hparams['train_mode']
        self.log_every = hparams['log_every']
        self.counter = 0
    def traing_log(self):
        if self.counter % self.log_every == 0:
            mean_loss = torch.tensor(self.all_loss).mean()
            self.writer.add_scalar('Loss/Train', mean_loss, global_step=self.counter)
            if self.train_mode == 'supervised':
                self.all_pred, self.all_label = torch.cat(self.all_pred), torch.cat(self.all_label)
                accuracy = (self.all_pred == self.all_label).sum() / len(self.all_pred) * 100
                self.writer.add_scalar('Accuracy/Train', accuracy, global_step=self.counter)
                self.all_pred = []
                self.all_label = []
            self.all_loss = []
    def val_log(self, val_loss = None, val_accuracy = None):
        if val_loss: self.writer.add_scalar('Loss/Validation', val_loss, global_step=self.counter)
        if val_accuracy: self.writer.add_scalar('Accuracy/Validation', val_accuracy, global_step=self.counter)
    def step(self):
        self.counter += 1





class largest_component(object):

    def crop(self, image): 
        y_nonzero, x_nonzero = np.nonzero(image)
        retval = image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        return retval
    
    def __call__(self, pic):
        return self.get_largest_component(pic)
    
    def get_largest_component(self, image):
        """
          get the largest component from 2D or 3D binary image
          image: nd array
        """
        image = np.array(image)
        # image = image[0]
        orig_image = np.copy(image)
        dim = len(image.shape)
        image[image < 0.1] = 0

        if(image.sum() == 0 ):
            print('the largest component is null')
            return image
        if(dim == 2):
            s = ndimage.generate_binary_structure(2,1)
        elif(dim == 3):               
            s = ndimage.generate_binary_structure(3,1)
        else:
            raise ValueError(f"the dimension number should be 2 or 3. got dimension of {dim}")

        labeled_array, numpatches = ndimage.label(image,s)

        sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
        max_label = np.where(sizes == sizes.max())[0] + 1

        output = np.asarray(labeled_array == max_label, np.uint8)
        output = np.where(output[:,:] == 1,  orig_image[:,:], 0)
        #cv2_imshow(self.crop(output))
        
        return output