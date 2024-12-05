import os
import torch
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.spatial import KDTree

# from .MILDataset import MILDataset
from .ProcessedMILDataset import ProcessedMILDataset

class PandaDataset(ProcessedMILDataset):
    def __init__(self, 
                 data_path, 
                 processed_data_path,
                 csv_path, 
                 aux_csv_path,
                 use_patch_distances = True,
                 n_samples = None,
                 discard_ambiguous = True, 
                 **kwargs
    ):
        self.extension = '.npy' if 'features' in data_path else '.jpg'
        self.dataset_name = 'PandaDataset'
        super(PandaDataset, self).__init__(
            processed_data_path, 
            data_path,
            csv_path=csv_path,
            aux_csv_path=aux_csv_path,
            n_samples=n_samples,
            discard_ambiguous=discard_ambiguous,
            use_patch_distances=use_patch_distances,
            **kwargs
        )

        if self.n_samples is not None:
            print(f'[{self.dataset_name}] Sampling {self.n_samples} bags...')
            self.bag_names = self.bag_names[:self.n_samples]
            self.data_dict = { k : self.data_dict[k] for k in self.bag_names }

    def _inst_loader(self, path):
        if path.endswith('.npy'):
            d = np.load(path)
        else:
            d = cv2.imread(path)
            d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
            d = d.transpose(2,0,1)
        return d

    def _init_data_dict(self):
        print(f'[{self.dataset_name}] Scanning files...')
        df = pd.read_csv(self.csv_path)

        def map_val(x):
            dic_label = {'NC' : 0, 'C': 1, 'unlabeled': -1}
            return dic_label[x]

        df['image_label'] = df[['NC', 'C', 'unlabeled']].idxmax(axis=1).apply(map_val)

        df['wsi_name'] = df['image_name'].apply(lambda x: x.split('_')[0])
        df.rename(columns={'image_label': 'inst_labels', 'wsi_name' : 'bag_name', 'image_name' : 'inst_names'}, inplace=True)
        df['inst_names'] = df['inst_names'].apply(lambda x: x.split('.')[0])

        grouped = df.groupby('bag_name')
        group_df = grouped['inst_labels'].apply(list).reset_index()
        group_df['inst_names'] = grouped['inst_names'].apply(list).reset_index()['inst_names']
        group_df['bag_label'] = grouped['inst_labels'].max().reset_index()['inst_labels']

        if self.discard_ambiguous:
            
            # discard WSIs labeled as ambiguous
            amiguous_idx = group_df['bag_label'] == -1
            group_df = group_df[~amiguous_idx]

            # discard WSIs such that the bag label is different from max(inst labels)
            aux_df = pd.read_csv(self.aux_csv_path)
            aux_df['bag_label'] = aux_df['isup_grade'].apply(lambda x : 0 if x == 0 else 1)
            aux_df['bag_name'] = aux_df['image_id']
            aux_df = aux_df[['bag_name', 'bag_label']]
            tmp_df = pd.merge(group_df, aux_df, on='bag_name', how='left', suffixes=('_old', '_new'))
            ambiguous_idx = tmp_df['bag_label_old'] != tmp_df['bag_label_new']
            group_df = group_df[~ambiguous_idx]

        data_dict = {}
        bag_names_list = group_df['bag_name'].values
        inst_labels_list = group_df['inst_labels'].values
        inst_names_list = group_df['inst_names'].values
        bag_labels_list = group_df['bag_label'].values

        num_wsis = len(bag_names_list)
        pbar = tqdm(range(num_wsis), total=num_wsis)
        for i in pbar:
            pbar.set_description(f'[{self.dataset_name}] Building data dict')
            bag_name = bag_names_list[i]
            inst_labels = inst_labels_list[i]
            inst_names = inst_names_list[i]
            inst_names = sorted(inst_names)

            len_inst_labels = len(inst_labels)
            len_inst_names = len(inst_names)

            if len_inst_labels != len_inst_names:
                print(f'[{self.dataset_name}] WARNING: {bag_name} has different number of labels and names: {len_inst_labels} vs {len_inst_names}')
                min_len = min(len_inst_labels, len_inst_names)
                inst_labels = inst_labels[:min_len]
                inst_names = inst_names[:min_len]
            
            if len(inst_labels) == 0:
                print(f'[{self.dataset_name}] WARNING: {bag_name} has no instances')
                continue

            inst_paths = [ os.path.join(self.data_path, s) + self.extension for s in inst_names]
            inst_coords = np.array([ (int(s.split('_')[1]), int(s.split('_')[2])) for s in inst_names ])
            bag_label = bag_labels_list[i]
            data_dict[bag_name] = {
                'inst_labels': np.array(inst_labels),
                'inst_names': inst_names,
                'inst_paths': inst_paths,
                'inst_coords': inst_coords,
                'inst_int_coords' : inst_coords.astype(int),
                'bag_label': bag_label,
            }
    
        return data_dict

    def _build_edge_index(self, coords, bag_feat):
        kdtree = KDTree(coords)

        # Build adjacency matrix
        n_patches = len(coords)
        edge_index = []
        edge_weight = []
        for i in range(n_patches):
            # Find neighboring patches within sqrt(2) distance
            neighbors = kdtree.query_ball_point(coords[i], np.sqrt(2))
            for j in neighbors:
                if i != j:
                    edge_index.append([i,j])
                    if self.use_patch_distances:
                        dist = np.exp( -np.linalg.norm(bag_feat[i] - bag_feat[j]) / bag_feat.shape[1] )
                    else:
                        dist = 1.0
                    edge_weight.append(dist)

        edge_index = np.array(edge_index).T.astype(np.longlong) # (2, n_edges)
        edge_weight = np.array(edge_weight) # (n_edges,)
        return edge_index, edge_weight
        
    def get_class_counts(self):
        class_counts = {}
        for bag_name in self.bag_names:
            bag_label = self.data_dict[bag_name]['bag_label']
            if bag_label not in class_counts:
                class_counts[bag_label] = 1
            else:
                class_counts[bag_label] += 1
        return class_counts