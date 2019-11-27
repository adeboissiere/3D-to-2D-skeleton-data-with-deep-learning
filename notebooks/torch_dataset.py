import h5py
import numpy as np
import random
import torch

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 samples_names_list):
        super(TorchDataset, self).__init__()
        
        self.data_path = data_path
        self.samples_names_list = samples_names_list
        
    def __getitem__(self, index):
        sample_name = self.samples_names_list[index]
        random_joint = random.randrange(25)
        
        # Get 3D joint
        with h5py.File(self.data_path + "skeleton.h5", 'r') as skeleton_3d_h5:
            random_frame = random.randrange(skeleton_3d_h5[sample_name]["skeleton"].shape[1])
            
            joint_3d = skeleton_3d = skeleton_3d_h5[sample_name]["skeleton"][:, random_frame, random_joint, 0]
            
        # Get 2D joint
        with h5py.File(self.data_path + "ir_skeleton.h5", 'r') as skeleton_3d_h5:
            joint_2d = skeleton_3d = skeleton_3d_h5[sample_name]["ir_skeleton"][:, random_frame, random_joint, 0]
            joint_2d[0] = joint_2d[0] / 512
            joint_2d[1] = joint_2d[1] / 424
            
            # Replace NaN with zero
            joint_2d[np.isnan(joint_2d)] = 0
            
        return joint_3d, joint_2d
    
    def __len__(self):
        return len(self.samples_names_list)
            