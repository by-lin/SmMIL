import os
import numpy as np
import pandas as pd

from .ProcessedMILDataset import ProcessedMILDataset

class RSNADataset(ProcessedMILDataset):
    def __init__(self, 
                 data_path,
                 processed_data_path,
                 csv_path, 
                 use_slice_distances=True,
                 n_samples=None, 
                 **kwargs
    ):
        self.dataset_name = 'RSNADataset'
        self.use_slice_distances = use_slice_distances
        self.n_samples = n_samples
        super(RSNADataset, self).__init__(
            processed_data_path = processed_data_path, 
            data_path = data_path,
            csv_path = csv_path,
            keep_in_memory = True, 
            **kwargs
        )

        print(f"[RSNADataset] Number of bags found: {len(self.bag_names)}")
        if self.n_samples is not None:
            print(f"[RSNADataset] Sampling {self.n_samples} bags...")
            #idx_vec = np.random.choice(len(bags_ids_dic), self.n_samples, replace=False)
            idx_vec = np.arange(len(self.bag_names))[0:self.n_samples]
            self.bag_names = [self.bag_names[i] for i in idx_vec]
            self.data_dict = {k: self.data_dict[k] for k in self.bag_names}
    
    def _inst_loader(self, path):
        if path.endswith('.npy'):
            d = np.load(path)
            if len(d.shape) == 3:
                d = d.transpose(2,0,1)
        else:
            raise ValueError(f"[RSNADataset] Unknown file format: {path}")
        return d
    
    def _init_data_dict(self):
        print("[RSNADataset] Scanning files...")
        df = pd.read_csv(self.csv_path)
        data_dict = {} # { bag_name: { 'bag_label': int, 'inst_paths' : [str, str, ...], 'inst_labels': array, 'L_mat' : array} }
        for bag_name in df.bag_name.unique():
            bag_dict = {} # { 'bag_label': int, 'inst_paths' : [str, str, ...], 'inst_labels': array, 'L_mat' : array}
            bag_df = df[df.bag_name==bag_name].sort_values(by=['order'])
            inst_names = [file.split('.')[0] for file in list(bag_df.instance_name)]
            bag_label = list(bag_df.bag_label)[0]
            inst_labels = []
            inst_paths = []
            inst_coords = []
            for i in range(len(inst_names)):
                name = inst_names[i] 
                path = os.path.join(self.data_path, name) + '.npy'
                if not os.path.exists(path):
                    print(f"[RSNADataset] Slice not found: {path}.") 
                else:
                    inst_paths.append(path)
                    inst_labels.append(list(bag_df.instance_label)[i])
                    inst_coords.append(i)
            bag_dict['inst_paths'] = inst_paths
            bag_dict['bag_label'] = float(bag_label)
            bag_dict['inst_labels'] = np.array(inst_labels).astype(np.float32)
            bag_dict['inst_coords'] = np.array(inst_coords).astype(np.float32)
            bag_dict['inst_int_coords'] = np.array(inst_coords).astype(np.int64)

            data_dict[bag_name] = bag_dict
        
        return data_dict

    def _build_edge_index(self, coords, bag_feat):
        n = len(coords)
        
        edge_index = []
        edge_weight = []

        for i in range(n):
            for j in [-1, 0, 1]:
                if i+j >= 0 and i+j < n:
                    edge_index.append([i, i+j])
                    if self.use_slice_distances:
                        dist = np.exp( - np.linalg.norm(bag_feat[i] - bag_feat[i+j]) / bag_feat.shape[1])
                    else:
                        dist = 1.0
                    edge_weight.append(dist)
        edge_index = np.array(edge_index).T.astype(np.longlong) # (2, n_edges)
        edge_weight = np.array(edge_weight) # (n_edges,)

        return edge_index, edge_weight
    
    def get_class_counts(self):
        class_counts = {}
        for bag_name in self.bag_names:
            label = self.data_dict[bag_name]['bag_label']
            if label not in class_counts.keys():
                class_counts[label] = 1
            else:
                class_counts[label] += 1
        return class_counts