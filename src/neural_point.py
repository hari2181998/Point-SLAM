import torch
import numpy as np

import faiss
import faiss.contrib.torch_utils
from src.common import setup_seed, clone_kf_dict


class NeuralPointCloud(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.c_dim = cfg['model']['c_dim']
        self.device = cfg['mapping']['device']
        self.cuda_id = 0
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.nn_num = cfg['pointcloud']['nn_num']

        self.nlist = cfg['pointcloud']['nlist']
        self.radius_add = cfg['pointcloud']['radius_add']
        self.radius_min = cfg['pointcloud']['radius_min']
        self.radius_query = cfg['pointcloud']['radius_query']
        self.fix_interval_when_add_along_ray = cfg['pointcloud']['fix_interval_when_add_along_ray']

        self.N_surface = cfg['rendering']['N_surface']
        self.N_add = cfg['pointcloud']['N_add']
        self.near_end_surface = cfg['pointcloud']['near_end_surface']
        self.far_end_surface = cfg['pointcloud']['far_end_surface']

        self._cloud_pos = []     # (input_pos) * N_add
        self._input_pos = []     # to save locations of the depth input
        self._input_rgb = []     # to save locations of the rgb input at the depth input
        self._pts_num = 0        # number of points in neural point cloud
        self.geo_feats = None
        self.col_feats = None
        self.keyframe_dict = []

        self.resource = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.resource,
                                            self.cuda_id,
                                            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
        self.index.nprobe = cfg['pointcloud']['nprobe']

        self.deform_points = cfg["deform_points"]
        if self.deform_points:
            self._pos_in_last_keyframe = []
            self._last_keyframe_for_pts = []
        
        setup_seed(cfg["setup_seed"])
    
    def pos_in_last_keyframe(self, index=None):
        if index is None:
            return self._pos_in_last_keyframe
        return self._pos_in_last_keyframe[index]
    
    def last_keyframe_for_pts(self, index=None):
        if index is None:
            return self._last_keyframe_for_pts
        return self._last_keyframe_for_pts[index]

    def cloud_pos(self, index=None):
        if index is None:
            return self._cloud_pos
        return self._cloud_pos[index]

    def input_pos(self):
        return self._input_pos

    def input_rgb(self):
        return self._input_rgb

    def pts_num(self):
        return self._pts_num

    def index_train(self, xb):
        assert torch.is_tensor(xb), 'use tensor to train FAISS index'
        self.index.train(xb)
        return self.index.is_trained

    def index_ntotal(self):
        return self.index.ntotal

    def get_radius_query(self):
        return self.radius_query

    def get_geo_feats(self):
        return self.geo_feats

    def get_col_feats(self):
        return self.col_feats

    def update_geo_feats(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self.geo_feats[indices] = feats.detach().clone()
        else:
            assert feats.shape[0] == self.geo_feats.shape[0], 'feature shape[0] mismatch'
            self.geo_feats = feats.detach().clone()

    def update_col_feats(self, feats, indices=None):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self.col_feats[indices] = feats.detach().clone()
        else:
            assert feats.shape[0] == self.col_feats.shape[0], 'feature shape[0] mismatch'
            self.col_feats = feats.detach().clone()

    def add_neural_points(self, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                          train=False, is_pts_grad=False, dynamic_radius=None,last_keyframe_idx =None, last_keyframe_c2w = None):
        """
        Add multiple neural points, will use depth filter when getting these samples.

        Args:
            batch_rays_o (tensor): ray origins (N,3)
            batch_rays_d (tensor): ray directions (N,3)
            batch_gt_depth (tensor): sensor depth (N,)
            batch_gt_color (tensor): sensor color (N,3)
            train (bool): whether to update the FAISS index
            is_pts_grad (bool): the points are chosen based on color gradient
            dynamic_radius (tensor): choose every radius differently based on its color gradient

        """

        if batch_rays_o.shape[0]:
            mask = batch_gt_depth > 0
            batch_gt_color = batch_gt_color*255
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                batch_rays_o[mask], batch_rays_d[mask], batch_gt_depth[mask], batch_gt_color[mask]

            pts_gt = batch_rays_o[..., None, :] + batch_rays_d[...,
                                                               None, :] * batch_gt_depth[..., None, None]
            mask = torch.ones(pts_gt.shape[0], device=self.device).bool()
            pts_gt = pts_gt.reshape(-1, 3)

            if self.index.is_trained:
                _, _, neighbor_num_gt = self.find_neighbors_faiss(
                    pts_gt, step='add', is_pts_grad=is_pts_grad, dynamic_radius=dynamic_radius)
                mask = (neighbor_num_gt == 0)

            self._input_pos.extend(pts_gt[mask].tolist())
            self._input_rgb.extend(batch_gt_color[mask].tolist())

            gt_depth_surface = batch_gt_depth.unsqueeze(
                -1).repeat(1, self.N_add)
            t_vals_surface = torch.linspace(
                0.0, 1.0, steps=self.N_add, device=self.device)

            if self.fix_interval_when_add_along_ray:
                # add along ray, interval unrelated to depth
                intervals = torch.linspace(-0.04, 0.04, steps=self.N_add,
                                           device=self.device).unsqueeze(0)
                z_vals = gt_depth_surface + intervals
            else:  # add along ray, interval related to depth
                z_vals_surface = self.near_end_surface*gt_depth_surface * (1.-t_vals_surface) + \
                    self.far_end_surface * \
                    gt_depth_surface * (t_vals_surface)
                z_vals = z_vals_surface

            pts = batch_rays_o[..., None, :] + \
                batch_rays_d[..., None, :] * z_vals[..., :, None]
            print("pts shape before masking: ", pts.shape)
            pts = pts[mask]  # use mask from pts_gt for auxiliary points
            print("pts shape after masking: ", pts.shape)
            pts = pts.reshape(-1, 3)

            self._cloud_pos += pts.tolist()
            self._pts_num += pts.shape[0]

            if self.deform_points:
                homo_pts = torch.cat([pts,torch.ones(pts.shape[0],1, device=self.device)],1 )
                last_keyframe_w2c = torch.inverse(last_keyframe_c2w)
                pts_in_last_keyframe = torch.matmul(last_keyframe_w2c, homo_pts.T).T
                # pts_in_last_keyframe = torch.cat([pts_in_last_keyframe, torch.ones(pts_in_last_keyframe.shape[0],1, device=self.device)],1)
                print("pts last keyframe: ", pts_in_last_keyframe.shape, pts.shape)
                self._pos_in_last_keyframe += pts_in_last_keyframe.tolist()
                self._last_keyframe_for_pts += [last_keyframe_idx] * len(pts_in_last_keyframe.tolist())


            if self.geo_feats is None:
                self.geo_feats = torch.zeros(
                    [self._pts_num, self.c_dim], device=self.device).normal_(mean=0, std=0.1)
                self.col_feats = torch.zeros(
                    [self._pts_num, self.c_dim], device=self.device).normal_(mean=0, std=0.1)
            else:
                self.geo_feats = torch.cat([self.geo_feats,
                                            torch.zeros([pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)], 0)
                self.col_feats = torch.cat([self.col_feats,
                                            torch.zeros([pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)], 0)

            if train or not self.index.is_trained:
                self.index.train(pts)
            self.index.train(torch.tensor(self._cloud_pos, device=self.device))
            self.index.add(pts)
            return torch.sum(mask)
        else:
            return 0

    def find_neighbors_faiss(self, pos, step='add', retrain=False, is_pts_grad=False, dynamic_radius=None):
        """
        Query neighbors using faiss.

        Args:
            pos (tensor): points to find neighbors
            step (str): 'add'|'query'
            retrain (bool, optional): if to retrain the index cluster of IVF
            is_pts_grad: whether it's the points chosen based on color grad, will use smaller radius when looking for neighbors
            dynamic_radius (tensor, optional): choose every radius differently based on its color gradient

        Returns:
            D: distances to neighbors for the positions in pos
            I: indices of neighbors for the positions in pos
            neighbor_num: number of neighbors for the positions in pos
        """
        if (not self.index.is_trained) or retrain:
            self.index.train(self._cloud_pos)

        assert step in ['add', 'query']
        split_pos = torch.split(pos, 65000, dim=0)
        D_list = []
        I_list = []
        for split_p in split_pos:
            D, I = self.index.search(split_p.float(), self.nn_num)
            D_list.append(D)
            I_list.append(I)
        D = torch.cat(D_list, dim=0)
        I = torch.cat(I_list, dim=0)

        if step == 'query':  # used if dynamic_radius is None
            radius = self.radius_query
        else:  # step == 'add', used if dynamic_radius is None
            if not is_pts_grad:
                radius = self.radius_add
            else:
                radius = self.radius_min

        # faiss returns "D" in the form of squared distances. Thus we compare D to the squared radius
        if dynamic_radius is not None:
            assert pos.shape[0] == dynamic_radius.shape[0], 'shape mis-match for input points and dynamic radius'
            neighbor_num = (D < dynamic_radius.reshape(-1, 1)
                            ** 2).sum(axis=-1).int()
        else:
            neighbor_num = (D < radius**2).sum(axis=-1).int()

        return D, I, neighbor_num

    def sample_near_pcl(self, rays_o, rays_d, near, far, num):
        """
        For pixels with 0 depth readings, preferably sample near point cloud.

        Args:
            rays_o (tensor): rays origin
            rays_d (tensor): rays direction
            near : near end for sampling along this ray
            far: far end
            num (int): sampling num between near and far

        Returns:
            z_vals (tensor): z values for zero valued depth pixels
            invalid_mask (bool): mask for zero valued depth pixels that are not close to neural point cloud
        """
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        n_rays = rays_d.shape[0]
        intervals = 25
        z_vals = torch.linspace(near, far, steps=intervals, device=self.device)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(-1, 3)

        if torch.is_tensor(far):
            far = far.item()
        z_vals_section = np.linspace(near, far, intervals)
        z_vals_np = np.linspace(near, far, num)
        z_vals_total = np.tile(z_vals_np, (n_rays, 1))

        pts_split = torch.split(pts, 65000)  # limited by faiss bug
        Ds, Is, neighbor_nums = [], [], []
        for pts_batch in pts_split:
            D, I, neighbor_num = self.find_neighbors_faiss(
                pts_batch, step='query')
            D, I, neighbor_num = D.cpu().numpy(), I.cpu().numpy(), neighbor_num.cpu().numpy()
            Ds.append(D)
            Is.append(I)
            neighbor_nums.append(neighbor_num)
        D = np.concatenate(Ds, axis=0)
        I = np.concatenate(Is, axis=0)
        neighbor_num = np.concatenate(neighbor_nums, axis=0)

        neighbor_num = neighbor_num.reshape((n_rays, -1))
        # a point is True if it has at least one neighbor
        neighbor_num_bool = neighbor_num.astype(bool)
        # a ray is invalid if it has less than two True points along the ray
        invalid = neighbor_num_bool.sum(axis=-1) < 2

        if invalid.sum(axis=-1) < n_rays:
            # select, for the valid rays, a subset of the 25 points along the ray (num points = 5)
            # that are close to the surface.
            r, c = np.where(neighbor_num[~invalid].astype(bool))
            idx = np.concatenate(
                ([0], np.flatnonzero(r[1:] != r[:-1])+1, [r.size]))
            out = [c[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
            z_vals_valid = np.asarray([np.linspace(
                z_vals_section[item[0]], z_vals_section[item[1]], num=num) for item in out])
            z_vals_total[~invalid] = z_vals_valid

        invalid_mask = torch.from_numpy(invalid).to(self.device)
        return torch.from_numpy(z_vals_total).float().to(self.device), invalid_mask

    def update_global_pos_for_keyframe(self,frame, c2w, last_keyframe_idx, last_keyframe_pos):
        """
        Update the global position of the keyframe points.
        """

        indices = last_keyframe_idx == frame
        # if no points in this keyframe, return
        if not torch.any(indices):
            return indices
        
        keyframe_pos = last_keyframe_pos[indices]
        global_pos = torch.matmul(c2w, keyframe_pos.T).T[:, :3]
        # print(global_pos.shape, keyframe_pos.shape, c2w.shape, "debugging update")
        cloud_pos_tensor = torch.tensor(self.cloud_pos(), device=self.device)
        cloud_pos_tensor[indices] = global_pos
        self._cloud_pos = cloud_pos_tensor.tolist()
        return indices

    def update_global_pos(self, cloud_pos):
        self._cloud_pos = cloud_pos.tolist()
        return
    
    def update_faiss_index(self):
        """
        Update the faiss index.
        """
        # self.index.update_vectors(len(indices), torch.tensor(indices, device=self.device), torch.tensor(self.cloud_pos(), device=self.device)[indices])
        self.index.reset()
        self.index.train(torch.tensor(self.cloud_pos(), device=self.device))
        self.index.add(torch.tensor(self.cloud_pos(), device=self.device))
        return 

    def update_keyframe_pos(self, frame, new_frame, c2w, last_keyframe_idx, cloud_pos, last_keyframe_pos):
        """
        Update the keyframe position of certain points
        """
    
        # indices = last_keyframe_idx == frame
        # # keyframe_pos = torch.tensor(self.pos_in_last_keyframe(), device=self.device)
        # indices_pos = last_keyframe_pos[indices]
        # w2c = torch.inverse(c2w)
        # new_pos = torch.matmul(w2c, indices_pos.T).T
        # last_keyframe_pos[indices] = new_pos
        # self._pos_in_last_keyframe = last_keyframe_pos.tolist()
        # last_keyframe_idx[indices] = new_frame
        # self._last_keyframe_for_pts = last_keyframe_idx.tolist()

        w2c = torch.inverse(c2w)
        # print("homo cloud")
        homo_cloud_pos = torch.cat([cloud_pos, torch.ones(cloud_pos.shape[0],1,device=cloud_pos.device)], dim=1)
        last_keyframe_pos = last_keyframe_pos.to(cloud_pos.device)
        last_keyframe_idx = last_keyframe_idx.to(cloud_pos.device)
        w2c = w2c.to(cloud_pos.device)
        # print(homo_cloud_pos.shape)
        last_keyframe_pos = torch.where(last_keyframe_idx.unsqueeze(1).to(cloud_pos.device)==frame, torch.einsum('ij,kj->ki',w2c, homo_cloud_pos), last_keyframe_pos)
        # print(last_keyframe_pos.shape)
        last_keyframe_idx = torch.where(last_keyframe_idx == frame, new_frame.to(cloud_pos.device), last_keyframe_idx)
        self._pos_in_last_keyframe = last_keyframe_pos.tolist()
        self._last_keyframe_for_pts = last_keyframe_idx.tolist()
        return