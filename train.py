# %%
import numpy as np
import pandas as pd 

import os

from encode import rle_encode
from vision_models import ResNetUNet

from dataset import ImageDataset, train_test_valid_split, broadcast, next_remainder
from dataset import gray_to_tiled_tensor, image_feature_tensor, label_feature_tensor
from torch.utils.data import Subset, DataLoader
import glob
from matplotlib import pyplot as plt 
import torch
from torch.optim import lr_scheduler, Adam
import copy
import time
import torch.nn.functional as F
from  losses import DiceLoss, IOU
from collections import defaultdict
import skimage
import gc
import subprocess
from metric import score
from functools import partial
from metric import compute_surface_dice_at_tolerance, compute_surface_distances
import multiprocessing

# %%


# %%
offline_mode = True

# %%
def load_resnet18_weigths():
    if not os.path.exists('/root/.cache/torch/hub/checkpoints/'):
        os.makedirs('/root/.cache/torch/hub/checkpoints/')
    source_path = "/kaggle/input/resnet18/resnet18.pth"
    destination_path = "/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    subprocess.run(["cp", source_path, destination_path])

# %%
if offline_mode:
    load_resnet18_weigths()

# %%


# %%
data_root = "/kaggle/input"

train_path = data_root  + "/blood-vessel-segmentation/train/"

source_path = train_path + "kidney_3_sparse/images/*" 

target_path = train_path + "kidney_3_dense/images/" 
test_path = data_root + "/blood-vessel-segmentation/test"



# %%

group_weights = [
    {"group" :"kidney_1_voi", "pos_weigth":1, "remove_background": False, "dilate_label": True, "image_files":[], "label_files":[]},
    {"group" :"kidney_1_dense","pos_weigth":1, "remove_background": True, "dilate_label": False, "image_files":[], "label_files":[]},
    {"group" :"kidney_3_dense","pos_weigth":1, "remove_background": False, "dilate_label": False, "image_files":[], "label_files":[]},
    {"group" :"kidney_2", "pos_weigth":0.85, "remove_background": False, "dilate_label": False, "image_files":[], "label_files":[]},
    {"group" :"kidney_3_sparse", "pos_weigth":0.65, "remove_background": False, "dilate_label": False, "image_files":[], "label_files":[]}
]
def get_files(folder):
    _files = list(filter(os.path.isfile, glob.glob(folder + "*.tif")))
    _files.sort()
    return _files

for item in group_weights:
    if item["group"] != "kidney_3_dense":
        image_files = get_files(train_path + item["group"] + "/images/")
    else:
        source_path = train_path + "kidney_3_sparse/images/"
        label_folder = train_path + "kidney_3_dense/labels/"
        label_files = glob.glob(os.path.join(label_folder, "*.tif"))  
        image_slices = [os.path.basename(label_file) for label_file in label_files] 
        image_files = []
        for image_slice in image_slices:
            image_files.append(os.path.join(source_path, image_slice))
        image_files.sort()
    item["image_files"] = image_files
    label_files = get_files(train_path + item["group"] + "/labels/")
    item["label_files"] = label_files


# %%
tile_width = 192
tile_height = 192
mini_batch_size = 64
num_class = 2



# %%
def collate_fn(data,dilate_label):
    im_list = []
    l_list = []
    desc_list = []
    for batch in data: 
        desc , im , l = batch
        desc_list.append(desc)
        im_list.append(image_feature_tensor(im,(tile_width,tile_height))) 
        l_list.append(label_feature_tensor(l,(tile_width,tile_height),dilate_label=dilate_label))
    return desc_list, torch.stack(im_list), torch.stack(l_list)



def train_val_kidney_dataloaders(image_files,label_files, batch_size = 1,test_split = 0.2, val_split= 0.1, dilate_label= True, remove_background= False):
    

    train_idx, test_idx, val_idx = train_test_valid_split(list(range(len(image_files))), test_size=test_split, valid_size=val_split)
    split_indices = {"train": train_idx, "test": test_idx, "val": val_idx}
    dataloaders = {} 
    # Iterate over different splits
    for split, idx in split_indices.items():
        split_image_files = list(Subset(image_files, idx))
        split_label_files = list(Subset(label_files, idx))

        dataset = ImageDataset(split_image_files,split_label_files,remove_background=remove_background)

        dataloaders[split] = DataLoader(dataset, batch_size=batch_size,collate_fn=partial(collate_fn,dilate_label=dilate_label),  shuffle=True, num_workers=multiprocessing.cpu_count()-1,pin_memory=True)
        
    return dataloaders


# %%

def collate(input_t, mini_batch_size=5):
    batch_size, n_rows, n_columns, canals, tile_w , tile_h  = input_t.shape
    flattened = input_t.reshape(batch_size*n_rows*n_columns,canals,tile_w,tile_h)
    batch_tiles = batch_size*n_rows*n_columns
    for i in range(0, batch_tiles, mini_batch_size):
        start = i
        end = min(i + mini_batch_size, batch_tiles)
        yield flattened[start:end]  
        

# %%

def uncollate(batch, original_shape, mini_batch_size= 5):
    batch_size, n_rows, n_columns, canals, tile_w , tile_h  = original_shape
    flattened = torch.zeros(batch_size*n_rows*n_columns,canals,tile_w,tile_h)
    for i, mini_batch in enumerate(batch):
        for ti, tile in enumerate(mini_batch):
            flattened[i*mini_batch_size + ti] = tile 
    prediction = flattened.reshape(original_shape)
    return prediction


# %%


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_class =2
model = ResNetUNet(num_class).to(device)


# %%

def calc_loss(pred, target, metrics, pos_weight = 1):

    bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor([pos_weight]).to(device))

    pred = F.sigmoid(pred)
    dice = DiceLoss()
    iou = IOU()
    dice_loss = dice(target,pred)*pred.shape[0]
    iou = iou(target,pred)*pred.shape[0]
    loss = dice_loss + bce*pred.shape[0]
    metrics["dice"] += dice_loss.detach()
    metrics["iou"] += iou.detach()
    metrics["bce"] += bce.detach()*pred.shape[0]
    metrics["loss"] +=  loss.detach()
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, dataloaders, optimizer,pos_weight = 1, num_epochs=1, mini_batch_size = 64):
    scaler = torch.cuda.amp.GradScaler()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    losses = {"train": [], "val": []}
    scores = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            #inc = 0
            metrics = defaultdict(float)
            epoch_samples = 0
            dl_count = 0
            
            optimizer.zero_grad()
            accumulation_steps = 10
            inc= 0
            for desc, inputs_tensor, labels_tensor in dataloaders[phase]:
            #for image, label in dataloaders[phase]:
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    label_tensor_shape = labels_tensor.shape
                    
                    mini_batch_predictions= []
                    # flattened_prediction 
                    for images, labels in zip(collate(inputs_tensor,mini_batch_size), collate(labels_tensor,mini_batch_size)):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):

                            outputs = model(images.to(device))
                            assert outputs.dtype is torch.float16
                            loss = calc_loss(outputs, labels.to(device), metrics,pos_weight)
                            losses[phase].append([metrics["dice"].cpu().detach()/epoch_samples,metrics["iou"].cpu().detach()/epoch_samples,metrics["loss"].cpu().detach()/epoch_samples])
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            if (inc+1) % accumulation_steps == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                        inc+=1
                        torch.cuda.memory_cached() 
                        epoch_samples += images.size(0)
                
                    epoch_loss = metrics['loss'] / epoch_samples
                with torch.no_grad():
                        torch.cuda.empty_cache()
                        gc.collect()
                
                
                if phase == 'val' and epoch_loss < best_loss:
                    
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                dl_count +=1   
        PATH  = "model"+ group_weights[0]["group"]+ "_e_"+ str(epoch) + ".pth"
        torch.save(best_model_wts,PATH)
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return losses, model


def apply_otsu(image):
            
    low , high = skimage.filters.threshold_multiotsu(image)
    thresh_mask = skimage.filters.apply_hysteresis_threshold(image,low, high )
    return thresh_mask
def plot_losses(losses,keys = ["dice","iou","loss"]):
    train_losses = np.array(losses["train"])
    val_losses = np.array(losses["val"])
    plt.figure()
    plt.title("train loss")

    
    for i in range(3):
        plt.plot(train_losses[:,i],label="train " + keys[i])
    plt.legend(loc='upper left')
    plt.figure()
    plt.title("val loss")
    for i in range(3):   
        plt.plot(val_losses[:,i], label= "validation " + keys[i])
    plt.legend(loc='upper left')
    plt.show()
    
def run(model, num_epochs , image_files, label_files,group , pos_weight =1 , batch_size = 2, mini_batch_size = 64, path_pretrained= None, lr= 1e-4, dilate_label= True, remove_background= False, plot = False):

    optimizer_ft = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    dataloaders = train_val_kidney_dataloaders(image_files,label_files,batch_size=2, test_split=0.1, val_split=0.4, dilate_label=dilate_label, remove_background=remove_background)
    if path_pretrained:
        model.load_state_dict(torch.load(path_pretrained))
    
    losses, model = train_model(model,dataloaders, optimizer_ft, pos_weight = pos_weight, num_epochs=num_epochs, mini_batch_size=mini_batch_size)
    plot_losses(losses)
    test_dl = dataloaders["test"]
    print("---model score in ----")
    print(group)
    print("score = ")
    print(model_score(model, test_dl, group, 24,plot))
    return model
def model_score(model, dataloader, group ,mini_batch_size, plot= False):
    threshold =0.5
    total_dice = 0 
    total_elements = 0 
    i = 0 
    model.eval()
    for  data in dataloader:
        desc, im_t, l_t = data
        mini_batch_predictions= []
        for mini_batch in collate(im_t,mini_batch_size):
            output = model(mini_batch.to(device))
            pred =  F.sigmoid(output)
            pred = pred.data.cpu() 
            mini_batch_predictions.append(pred)
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.memory_cached()
        prediction_t = uncollate(mini_batch_predictions,l_t.shape,mini_batch_size) 
        predictions= []
        n  = im_t.size(0)
        total_elements += n
        
        for j in range(n):
            canal1= broadcast(prediction_t[j][:,:,0,:,:])
            canal2= broadcast(prediction_t[j][:,:,0,:,:])
            image = broadcast(im_t[j][:,:,0,:,:])
            threshold_prediction = canal1 > threshold
            mask_gt = broadcast(l_t[j][:,:,0,:,:]).astype("bool")
            surface_dist = compute_surface_distances(
                mask_gt=mask_gt.astype("bool"),
                mask_pred=threshold_prediction.astype("bool"),
                spacing_mm=(1,1),
            )
            dice = compute_surface_dice_at_tolerance(
                surface_dist,
                tolerance_mm=0.0,
            )
            total_dice += np.nan_to_num(dice)
            if plot == True and  i <= 2:
                fig, axs = plt.subplots( 1,4, figsize=(12,4))
                
                axs[0].imshow(image)
                axs[0].axis('off')
                axs[1].set_title("ground truth")
                axs[1].imshow(mask_gt)
                axs[1].axis('off')
                axs[2].set_title("prediction")
                im = axs[2].imshow(canal1)
                axs[3].imshow(canal2)
                axs[3].axis('off')
                plt.tight_layout()
                cbar = fig.colorbar(im, ax=axs, orientation='vertical')
                cbar.set_label('Intensity') 
        i+=1  
    return total_dice/total_elements

# %%


# %%

num_epochs = 6

for params in group_weights:
    group = params["group"]
    print("training group ")
    print(group)
    files = [train_path + group + "/images/",train_path + group + "/labels/" ]
    model = run(model,num_epochs,params["image_files"],params["label_files"], group,params["pos_weigth"], dilate_label=params["dilate_label"], remove_background=params["remove_background"],batch_size=2,mini_batch_size=mini_batch_size,plot= False)


# %%
def predict(model, image):
    mini_batch_size = 24
    w,h = image.shape
    image_equalized = skimage.exposure.equalize_adapthist(image)
    im_t = image_feature_tensor(image_equalized*256,(tile_width,tile_height))
    reshaped_tensor = im_t.reshape(1, *im_t.shape)
    mini_batch_predictions= []
    for mini_batch in collate(reshaped_tensor,mini_batch_size):
        output = model(mini_batch.to(device))
        pred =  F.sigmoid(output)
        pred = pred.data.cpu() 
        mini_batch_predictions.append(pred)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.memory_cached()
    shape = np.array(reshaped_tensor.shape)
    shape[3] = num_class
    prediction_t = uncollate(mini_batch_predictions,tuple(shape),mini_batch_size) 
    
    canal1= broadcast(prediction_t[0][:,:,0,:,:])
    #canal2= broadcast(prediction_t[0][:,:,0,:,:])
    return canal1[:w,:h].astype(np.uint8)


# %%
# load best model weights 
# 

# %%
ids = []
rle_masks = []
threshold = 0.5
contents = os.listdir(test_path)
dataloaders = {}

for item in os.listdir(test_path):
    if os.path.isdir(os.path.join(test_path, item)):
        content_path = os.path.join(test_path, item)
        image_files = get_files(content_path + "/images/")
        for i, image_file in enumerate(image_files):
            image = skimage.io.imread(image_file)
            prediction = predict(model,image)
            thresh_prediction = (prediction>threshold).astype("bool")
            slice_number = (os.path.basename(image_file)).split(".")[0]#image_file.split('.')[0].split("/")[-1]
            ids.append(item + "_" + (slice_number))
            encoded_image = rle_encode(thresh_prediction)
            if encoded_image == "":
                rle_masks.append("1 0")
            else:
                rle_masks.append(encoded_image)

            
submission = pd.DataFrame({'id': ids, 'rle': rle_masks})
submission.to_csv("submission.csv",index=False)   


