import os
import os.path as osp
import numpy as np
import torch
import copy
from human_models.human_models import SMPL, SMPLX
from utils.data_utils import load_img, process_bbox, augmentation, \
    process_db_coord, process_human_model_output, \
    process_db_coord_crop, gen_cropped_hands
from utils.transforms import rigid_align, batch_rodrigues
import tqdm
import time
import random
import pickle
from constants import *




class Cache():
    """ A custom implementation for OSX pipeline
        Need to run tool/cache/fix_cache.py to fix paths
    """
    def __init__(self, load_path=None):
        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
        self.load_path = load_path
        self.cache = np.load(load_path, allow_pickle=True)
        self.data_len = self.cache['data_len']
        self.data_strategy = self.cache['data_strategy']
        assert self.data_len == len(self.cache) - 2  # data_len, data_strategy
        self.cache = None

    @classmethod
    def save(cls, save_path, data_list, data_strategy):
        assert save_path is not None, 'save_path is None'
        data_len = len(data_list)
        cache = {}
        for i, data in enumerate(data_list):
            cache[str(i)] = data
        assert len(cache) == data_len
        # update meta
        cache.update({
            'data_len': data_len,
            'data_strategy': data_strategy})

        np.savez_compressed(save_path, **cache)
        print(f'Cache saved to {save_path}.')

    # def shuffle(self):
    #     random.shuffle(self.mapping)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.cache is None:
            self.cache = np.load(self.load_path, allow_pickle=True)
        # mapped_idx = self.mapping[idx]
        # cache_data = self.cache[str(mapped_idx)]
        cache_data = self.cache[str(idx)]
        data = cache_data.item()
        return data


class HumanDataset(torch.utils.data.Dataset):

    def __init__(self, transform, data_split, cfg):
        self.transform = transform
        self.data_split = data_split
        self.cfg = cfg

        # dataset information, to be filled by child class
        self.img_dir = None
        self.annot_path = None
        self.annot_path_cache = None
        self.use_cache = False
        self.save_idx = 0
        self.img_shape = None  # (h, w)
        self.cam_param = None  # {'focal_length': (fx, fy), 'princpt': (cx, cy)}
        self.use_betas_neutral = False

        self.smpl_x = SMPLX.get_instance()
        self.smpl = SMPL.get_instance()

        self.joint_set = {
            'joint_num': self.smpl_x.joint_num,
            'joints_name': self.smpl_x.joints_name,
            'flip_pairs': self.smpl_x.flip_pairs}
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')

        self.downsample_mat = pickle.load(open(f'{self.cfg.model.human_model_path}/smplx2smpl.pkl', 
                                                'rb'))['matrix']

    def load_cache(self, annot_path_cache):
        datalist = Cache(annot_path_cache)
        return datalist

    def save_cache(self, annot_path_cache, datalist):
        print(f'[{self.__class__.__name__}] Caching datalist to {self.annot_path_cache}...')
        Cache.save(
            annot_path_cache,
            datalist,
            data_strategy=getattr(self.cfg.data, 'data_strategy', None)
        )

    def load_data(self, train_sample_interval=1, test_sample_interval=1):

        content = np.load(self.annot_path, allow_pickle=True)
        num_examples = len(content['image_path'])
        
        if 'meta' in content:
            meta = content['meta'].item()
            print('meta keys:', meta.keys())
            if 'annot_valid' in meta.keys(): # agora
                annot_valid = meta['annot_valid']
            else: 
                annot_valid = None

            if 'valid_label' in meta.keys(): # Ubody
                invalid_label = np.array(meta['valid_label']) == 0 # skip when True
                iscrowd = np.array(meta['iscrowd']) # skip when True
                num_keypoints_zero = np.array(meta['num_keypoints']) == 0 # skip when True

                skip_ubody = [iscrowd[i] or num_keypoints_zero[i] or invalid_label[i] for i in range(len(iscrowd))]
            else: 
                skip_ubody = None
            
            if 'iscrowd' in meta.keys(): # mscoco
                iscrowd = np.array(meta['iscrowd']) # skip when True
                num_keypoints_zero = np.array(meta['num_keypoints']) == 0 # skip when True

                skip_mscoco = [iscrowd[i] or num_keypoints_zero[i] for i in range(len(iscrowd))]
            else: 
                skip_mscoco = None

        else:
            meta = None
            annot_valid = None
            skip_ubody = None
            skip_mscoco= None
            print('No meta info provided! Please give height and width manually')

        #  ARCTIC val set
        if 'vertices3d_path' in content:
            vertices3d_path = content['vertices3d_path']
        else:
            vertices3d_path = None

        print(f'Start loading humandata {self.annot_path} into memory...\nDataset includes: {content.files}'); tic = time.time()
        image_path = content['image_path']

        if meta is not None and 'height' in meta:
            height = np.array(meta['height'])
            width = np.array(meta['width'])
            image_shape = np.stack([height, width], axis=-1)
        else:
            image_shape = None

        if self.__class__.__name__ == 'Hi4D':
            image_shape = None


        if 'smplx' in content:
            smplx = content['smplx'].item()
            as_smplx = 'smplx'
            if self.__class__.__name__ == 'UBody':
                smplx.pop('leye_pose')
                smplx.pop('reye_pose')
        elif 'smpl' in content:
            smplx = content['smpl'].item()
            as_smplx = 'smpl'
        elif 'smplh' in content:
            smplx = content['smplh'].item()
            as_smplx = 'smplh'

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__ == 'SHAPY':
            smplx = {}

        else:
            raise KeyError('No SMPL for SMPLX available, please check keys:\n'
                        f'{content.files}')
        
        if self.__class__.__name__ == 'PW3D' and 'test' in self.annot_path:
            print('load smpl for PW3d!')
            smplx = content['smpl'].item()
            as_smplx = 'smpl'
            gender = content['meta'].item()['gender']
        else:
            gender = None

        print('Smplx param', smplx.keys())

        # mano
        if 'mano' in content:
            mano = content['mano']
        else:
            mano = None

        # bbox
        if 'bbox_xywh' in content:
            bbox_xywh = content['bbox_xywh']
        else:
            raise KeyError(f'Necessary key [bbox_xywh] is missing in HumanData for {self.__class__.__name__}.')

        if 'lhand_bbox_xywh' in content:
            lhand_bbox_xywh = content['lhand_bbox_xywh']
        else:
            lhand_bbox_xywh = np.zeros((num_examples, 5))

        if 'rhand_bbox_xywh' in content:
            rhand_bbox_xywh = content['rhand_bbox_xywh']
        else:
            rhand_bbox_xywh = np.zeros((num_examples, 5))

        if 'face_bbox_xywh' in content:
            face_bbox_xywh = content['face_bbox_xywh']
        else:
            face_bbox_xywh = np.zeros((num_examples, 5))

        decompressed = False
        if content['__keypoints_compressed__']:
            decompressed_kps = self.decompress_keypoints(content)
            decompressed = True

        keypoints3d = None
        valid_kps3d = False
        keypoints3d_mask = None
        valid_kps3d_mask = False
        for kps3d_key in KPS3D_KEYS:
            if kps3d_key in content:
                keypoints3d = decompressed_kps[kps3d_key][:, SMPLX_137_MAPPING, :3] if decompressed \
                else content[kps3d_key][:, SMPLX_137_MAPPING, :3]
                valid_kps3d = True

                if f'{kps3d_key}_mask' in content:
                    keypoints3d_mask = content[f'{kps3d_key}_mask'][SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                elif 'keypoints3d_mask' in content:
                    keypoints3d_mask = content['keypoints3d_mask'][SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                break

        for kps2d_key in KPS2D_KEYS:
            if kps2d_key in content:
                keypoints2d = decompressed_kps[kps2d_key][:, SMPLX_137_MAPPING, :2] if decompressed \
                    else content[kps2d_key][:, SMPLX_137_MAPPING, :2]

                if f'{kps2d_key}_mask' in content:
                    keypoints2d_mask = content[f'{kps2d_key}_mask'][SMPLX_137_MAPPING]
                elif 'keypoints2d_mask' in content:
                    keypoints2d_mask = content['keypoints2d_mask'][SMPLX_137_MAPPING]
                break

        mask = keypoints3d_mask if valid_kps3d_mask \
                else keypoints2d_mask

        print('Done. Time: {:.2f}s'.format(time.time() - tic))

        datalist = []
        
        for i in tqdm.tqdm(range(int(num_examples))):
            if annot_valid is not None and not annot_valid[i]: continue # for agora
            if skip_ubody is not None and skip_ubody[i]: continue # for ubody
            if skip_mscoco is not None and skip_mscoco[i]: continue # for mscoco

            if self.data_split == 'train' and i % train_sample_interval != 0:
                continue
            if self.data_split == 'test' and i % test_sample_interval != 0:
                continue

            if vertices3d_path is not None:
                vertices3d = np.load(osp.join(self.img_dir, vertices3d_path[i]))
            else:
                vertices3d = None
            
            if 'MPI_INF_3DHP' in self.__class__.__name__:
                img_path = osp.join(self.img_dir, image_path[i][1:]) # remove the first /
            else:
                img_path = osp.join(self.img_dir, image_path[i])

            # import pdb; pdb.set_trace()
            img_shape = image_shape[i] if image_shape is not None else self.img_shape

            joint_img = keypoints2d[i]
            joint_valid = mask.reshape(-1, 1)

            bbox = bbox_xywh[i][:4]
            lhand_bbox = lhand_bbox_xywh[i]
            rhand_bbox = rhand_bbox_xywh[i]
            face_bbox = face_bbox_xywh[i]
            if hasattr(self.cfg.data, 'bbox_ratio'):
                bbox_ratio = self.cfg.data.bbox_ratio * 0.833 # preprocess body bbox is giving 1.2 box padding
            else:
                bbox_ratio = 1.25
            left_hand_chosen = None

            bbox = process_bbox(bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=bbox_ratio, 
                                input_img_shape=self.cfg.model.input_img_shape)
            if bbox is None: 
                print("skip since no bbox")
                continue
            # if hasattr(cfg, 'do_crop'):
            #     if cfg.do_crop: 
            #         joint_valid_temp = process_db_coord_crop(bbox, joint_img) 

            if lhand_bbox[-1] > 0:  # conf > 0
                lhand_bbox = lhand_bbox[:4]
                if hasattr(self.cfg.data, 'bbox_ratio'):
                    lhand_bbox = process_bbox(lhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=self.cfg.data.bbox_ratio,
                                            input_img_shape=self.cfg.model.input_img_shape)
                if lhand_bbox is not None:
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
            else:
                lhand_bbox = None
            if rhand_bbox[-1] > 0:
                rhand_bbox = rhand_bbox[:4]
                if hasattr(self.cfg.data, 'bbox_ratio'):
                    rhand_bbox = process_bbox(rhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=self.cfg.data.bbox_ratio,
                                            input_img_shape=self.cfg.model.input_img_shape)
                if rhand_bbox is not None:
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
            else:
                rhand_bbox = None
            if face_bbox[-1] > 0:
                face_bbox = face_bbox[:4]
                if hasattr(self.cfg.data, 'bbox_ratio'):
                    face_bbox = process_bbox(face_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=self.cfg.data.bbox_ratio,
                                            input_img_shape=self.cfg.model.input_img_shape)
                if face_bbox is not None:
                    face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
            else:
                face_bbox = None

            if valid_kps3d:
                joint_cam = keypoints3d[i]
            else:
                joint_cam = None

            smplx_param = {k: v[i] for k, v in smplx.items()}
            
            # agora skip kids
            is_kids = smplx_param.pop('betas_extra', 0)
            # import pdb; pdb.set_trace()
            if is_kids != 0:
                print('skip kids')
                continue

            # TODO: set invalid if None?
            smplx_param['body_pose'] = smplx_param.pop('body_pose', None)
            smplx_param['root_pose'] = smplx_param.pop('global_orient', None)
            smplx_param['shape'] = smplx_param.pop('betas', np.zeros(10, dtype=np.float32))
            smplx_param['shape'] = smplx_param['shape'][:10]    
            smplx_param['trans'] = smplx_param.pop('transl', np.zeros(3))
            smplx_param['lhand_pose'] = smplx_param.pop('left_hand_pose', None)
            smplx_param['rhand_pose'] = smplx_param.pop('right_hand_pose', None)
            smplx_param['expr'] = smplx_param.pop('expression', None)

            # TODO do not fix betas, give up shape supervision
            if 'betas_neutral' in smplx_param:
                smplx_param['shape'] = smplx_param.pop('betas_neutral')
                # smplx_param['shape'] = np.zeros(10, dtype=np.float32)
                smplx_param['shape'] = smplx_param['shape'][:10]
            
            # # TODO fix shape of poses
            if self.__class__.__name__ == 'Talkshow':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(21, 3)
                smplx_param['lhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['rhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['expr'] = smplx_param['expr'][:10]  
            
            if self.__class__.__name__ == 'ARCTIC':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) 

            # 'BEDLAM'
            if self.__class__.__name__ in ['GTA_Human2','GTA_Human_full', 
                                        'SynBody_whac', 'SynBody_Magic1','SynBody', 'SynBody_full', 'SynHand',
                                        'CHI3D', 'FIT3D', 'HumanSC3D', 
                                        'MOYO', 'ARCTIC',]:
                smplx_param['shape'] = smplx_param['shape'][:10]
                # print('[Flat Hand Mean]:manually set flat_hand_mean = True -> flat_hand_mean = False')
                # manually set flat_hand_mean = True -> flat_hand_mean = False
                smplx_param['lhand_pose'] -= HANDS_MEAN_L
                smplx_param['rhand_pose'] -= HANDS_MEAN_R


            if as_smplx == 'smpl':
                smplx_param['smpl_pose'] = smplx_param['body_pose']
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(-1, 3)
                smplx_param['body_pose'] = smplx_param['body_pose'][:21, :] # use smpl body_pose on smplx

                smplx_param['smpl_shape'] = smplx_param['shape']
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx
                
                if gender is not None:
                    smplx_param['gender'] = gender[i]

            if as_smplx == 'smplh':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx

            import pdb
            # for hand datasets, set shape and pose to all zero
            if self.__class__.__name__ in ['FreiHand', 'InterHand', 'BlurHand', 'HanCo']:
                smplx_param['shape'] = np.zeros((10, ))
                smplx_param['root_pose'] = np.zeros((3))
                smplx_param['body_pose'] = np.zeros((21, 3)) 
        
            if smplx_param['lhand_pose'] is None or (smplx_param['lhand_pose'] == 0).all():
                smplx_param['lhand_valid'] = False
                # TODO: manually set joint_valid to 0
                joint_valid[self.smpl_x.joint_part['lhand'], :] = 0
                joint_valid[self.smpl_x.lwrist_idx, :] = 0
            else:
                smplx_param['lhand_valid'] = True
                joint_valid[self.smpl_x.joint_part['lhand'], :] = 1
                joint_valid[self.smpl_x.lwrist_idx, :] = 1

            if smplx_param['rhand_pose'] is None or (smplx_param['rhand_pose'] == 0).all():
                smplx_param['rhand_valid'] = False
                joint_valid[self.smpl_x.joint_part['rhand'], :] = 0
                joint_valid[self.smpl_x.rwrist_idx, :] = 0
            else:
                smplx_param['rhand_valid'] = True
                joint_valid[self.smpl_x.joint_part['rhand'], :] = 1
                joint_valid[self.smpl_x.rwrist_idx, :] = 1
            
            if smplx_param['expr'] is None:
                smplx_param['face_valid'] = False
            else:
                smplx_param['face_valid'] = True

            if joint_cam is not None and np.any(np.isnan(joint_cam)):
                print("skip since no kps")
                continue
                
            sample = {
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'lhand_bbox': lhand_bbox,
                'rhand_bbox': rhand_bbox,
                'face_bbox': face_bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param,
                'model': as_smplx,
                'extrinsic_r': extrinsic_r[i] if 'extrinsic_r' in locals() else np.eye(3,3),
                'vertices3d': vertices3d if vertices3d is not None else -1,
                'idx': i}
            # Carry temporal identity fields when present (used by WorldPoseTemporal to
            # build per-(sequence, person) windows). These ride inside the npz 'meta' object.
            if meta is not None and 'person_id' in meta:
                sample['person_id'] = int(meta['person_id'][i])
                sample['frame_idx'] = int(meta['frame_idx'][i])
                sample['seq_name'] = str(meta['seq_name'][i])
            datalist.append(sample)

        # save memory
        del content, image_path, bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh, face_bbox_xywh, keypoints3d, keypoints2d

        if self.data_split == 'train':
            print(f'[{self.__class__.__name__} train] original size:', int(num_examples),
                  '. Sample interval:', train_sample_interval,
                  '. Sampled size:', len(datalist))

        if (getattr(self.cfg.data, 'data_strategy', None) == 'balance' and self.data_split == 'train') or \
                (getattr(self.cfg.data, 'data_strategy', None) == 'weighted' and self.data_split == 'train'):
            print(f'[{self.__class__.__name__}] Using [balance/weighted] strategy with datalist shuffled...')
            random.seed(2023)
            random.shuffle(datalist)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def _build_train_targets(self, data, img, img2bb_trans, bb2img_trans, rot, do_flip, img_shape):
        """Build (inputs, targets, meta_info) for a train sample given an already-augmented img.

        Shared by the single-frame __getitem__ and the temporal dataset (which replays the
        same augmentation params across the window and supervises only the center frame).
        """
        # h36m gt
        joint_cam = data['joint_cam']
        if joint_cam is not None:
            dummy_cord = False
            joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
        else:
            # dummy cord as joint_cam
            dummy_cord = True
            joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

        joint_img = data['joint_img']
        joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
        if not dummy_cord:
            joint_img[:, 2] = (joint_img[:, 2] / (self.cfg.model.body_3d_size / 2) + 1) / 2. * self.cfg.model.output_hm_shape[0]  # discretize depth

        joint_img_aug, joint_cam_wo_ra, \
        joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                                                    joint_img=joint_img,
                                                    joint_cam=joint_cam,
                                                    joint_valid=data['joint_valid'],
                                                    do_flip=do_flip,
                                                    img_shape=img_shape,
                                                    flip_pairs=self.joint_set['flip_pairs'],
                                                    img2bb_trans=img2bb_trans,
                                                    rot=rot,
                                                    src_joints_name=self.joint_set['joints_name'],
                                                    target_joints_name=self.smpl_x.joints_name,
                                                    input_img_shape=self.cfg.model.input_img_shape,
                                                    output_hm_shape=self.cfg.model.output_hm_shape,
                                                    input_body_shape=self.cfg.model.input_body_shape)

        # smplx coordinates and parameters
        smplx_param = data['smplx_param']
        smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
        smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, \
        smplx_mesh_cam_orig = process_human_model_output(
                                    human_model_param=smplx_param,
                                    cam_param=self.cam_param,
                                    do_flip=do_flip,
                                    img_shape=img_shape,
                                    img2bb_trans=img2bb_trans,
                                    rot=rot,
                                    human_model_type='smplx',
                                    joint_img=None if self.cam_param else joint_img,
                                    body_3d_size=self.cfg.model.body_3d_size,
                                    hand_3d_size=self.cfg.model.hand_3d_size,
                                    face_3d_size=self.cfg.model.face_3d_size,
                                    input_img_shape=self.cfg.model.input_img_shape,
                                    output_hm_shape=self.cfg.model.output_hm_shape,
                                    )

        # TODO temp fix keypoints3d for renbody
        if 'RenBody' in self.__class__.__name__:
            joint_cam_ra = smplx_joint_cam.copy()
            joint_cam_wo_ra = smplx_joint_cam.copy()
            joint_cam_wo_ra[self.smpl_x.joint_part['lhand'], :] = joint_cam_wo_ra[self.smpl_x.joint_part['lhand'], :] \
                                                            + joint_cam_wo_ra[self.smpl_x.lwrist_idx, None, :]  # left hand root-relative
            joint_cam_wo_ra[self.smpl_x.joint_part['rhand'], :] = joint_cam_wo_ra[self.smpl_x.joint_part['rhand'], :] \
                                                            + joint_cam_wo_ra[self.smpl_x.rwrist_idx, None, :]  # right hand root-relative
            joint_cam_wo_ra[self.smpl_x.joint_part['face'], :] = joint_cam_wo_ra[self.smpl_x.joint_part['face'], :] \
                                                            + joint_cam_wo_ra[self.smpl_x.neck_idx, None,: ]  # face root-relative
        # change smplx_shape if use_betas_neutral
        # processing follows that in process_human_model_output
        if self.use_betas_neutral:
            smplx_shape = smplx_param['betas_neutral'].reshape(1, -1)
            smplx_shape[(np.abs(smplx_shape) > 3).any(axis=1)] = 0.
            smplx_shape = smplx_shape.reshape(-1)

        # SMPLX pose parameter validity
        smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 9)).reshape(-1)
        smplx_joint_valid = smplx_joint_valid[:, None]
        smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc
        if not (smplx_shape == 0).all():
            smplx_shape_valid = True
        else:
            smplx_shape_valid = False

        # hand and face bbox transform
        lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans,
                                                        self.cfg.model.input_img_shape, self.cfg.model.output_hm_shape)
        rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans,
                                                        self.cfg.model.input_img_shape, self.cfg.model.output_hm_shape)
        face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape, img2bb_trans,
                                                        self.cfg.model.input_img_shape, self.cfg.model.output_hm_shape)
        if do_flip:
            lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
        rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
        face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
        rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
        face_bbox_size = face_bbox[1] - face_bbox[0]

        joint_img_aug = np.nan_to_num(joint_img_aug, nan=0.0)
        smplx_pose = np.nan_to_num(smplx_pose, nan=0.0)
        joint_cam_wo_ra = np.nan_to_num(joint_cam_wo_ra, nan=0.0)
        joint_cam_ra = np.nan_to_num(joint_cam_ra, nan=0.0)

        smplx_cam_trans = np.array(smplx_param['trans']) if 'trans' in smplx_param else None
        inputs = {'img': img}
        targets = {'joint_img': joint_img_aug, # keypoints2d
                   'smplx_joint_img': joint_img_aug, #smplx_joint_img, # projected smplx if valid cam_param, else same as keypoints2d
                   'joint_cam': joint_cam_wo_ra, # joint_cam actually not used in any loss, # raw kps3d probably without ra
                   'smplx_joint_cam': joint_cam_ra, # kps3d with body, face, hand ra # smplx_joint_cam if (dummy_cord or getattr(cfg, 'debug', False)) else
                   'smplx_pose': smplx_pose,
                   'smplx_shape': smplx_shape,
                   'smplx_expr': smplx_expr,
                   'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size,
                   'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size,
                   'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size,
                   'lhand_root': smplx_param['lhand_root'] if 'lhand_root' in smplx_param else np.zeros((1, 3)),
                   'rhand_root': smplx_param['rhand_root'] if 'rhand_root' in smplx_param else np.zeros((1, 3)),
                   'smplx_cam_trans': smplx_cam_trans}
        meta_info = {'joint_valid': joint_valid,
                     'joint_trunc': joint_trunc,
                     'smplx_joint_valid': smplx_joint_valid if dummy_cord else joint_valid,
                     'smplx_joint_trunc': smplx_joint_trunc if dummy_cord else joint_trunc,
                     'smplx_pose_valid': smplx_pose_valid,
                     'smplx_shape_valid': float(smplx_shape_valid),
                     'smplx_expr_valid': float(smplx_expr_valid),
                     'is_3D': float(False) if dummy_cord else float(True),
                     'lhand_bbox_valid': lhand_bbox_valid,
                     'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid,
                     }

        return inputs, targets, meta_info

    def __getitem__(self, idx):
        try:
            data = copy.deepcopy(self.datalist[idx])
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            exit(0)

        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        img = load_img(img_path)
        no_aug = getattr(self.cfg.data, 'no_aug', False)
        img, img2bb_trans, bb2img_trans, rot, do_flip, _scale, _color_scale = augmentation(
                                                                    no_aug, img, bbox,
                                                                    self.data_split,
                                                                    self.cfg.model.input_img_shape)
        img = self.transform(img.astype(np.float32)) / 255.

        ## for vis on original img
        focal = [self.cfg.model.focal[0] / self.cfg.model.input_body_shape[1] * bbox[2], 
                self.cfg.model.focal[1] / self.cfg.model.input_body_shape[0] * bbox[3]]
        princpt = [self.cfg.model.princpt[0] / self.cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                self.cfg.model.princpt[1] / self.cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

        if self.data_split == 'train':
            return self._build_train_targets(data, img, img2bb_trans, bb2img_trans,
                                             rot, do_flip, img_shape)

        # test
        else:
            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            if not dummy_cord:
                joint_img[:, 2] = (joint_img[:, 2] / (self.cfg.model.body_3d_size / 2) + 1) / 2. * self.cfg.model.output_hm_shape[0]  # discretize depth

            joint_img, joint_cam, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                                                        joint_img=joint_img,
                                                        joint_cam=joint_cam, 
                                                        joint_valid=data['joint_valid'], 
                                                        do_flip=do_flip, 
                                                        img_shape=img_shape, 
                                                        flip_pairs=self.joint_set['flip_pairs'], 
                                                        img2bb_trans=img2bb_trans, 
                                                        rot=rot,
                                                        src_joints_name=self.joint_set['joints_name'], 
                                                        target_joints_name=self.smpl_x.joints_name, 
                                                        input_img_shape=self.cfg.model.input_img_shape, 
                                                        output_hm_shape=self.cfg.model.output_hm_shape, 
                                                        input_body_shape=self.cfg.model.input_body_shape)
            
            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            smplx_cam_trans = np.array(smplx_param['trans']) if 'trans' in smplx_param else None

            model_type = data['model']
            if model_type == 'smplx':
                smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
                smplx_pose_valid, smplx_joint_valid, \
                    smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                                                                human_model_param=smplx_param, 
                                                                cam_param=self.cam_param, 
                                                                do_flip=do_flip, 
                                                                img_shape=img_shape, 
                                                                img2bb_trans=img2bb_trans,
                                                                rot=rot, 
                                                                human_model_type=model_type, 
                                                                joint_img=None if self.cam_param else joint_img,
                                                                body_3d_size=self.cfg.model.body_3d_size, 
                                                                hand_3d_size=self.cfg.model.hand_3d_size, 
                                                                face_3d_size=self.cfg.model.face_3d_size,
                                                                input_img_shape=self.cfg.model.input_img_shape, 
                                                                output_hm_shape=self.cfg.model.output_hm_shape, 
                                                                )
                smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 9)).reshape(-1)

            elif model_type == 'smpl':
                    _, _, _, _, _, smplx_mesh_cam_orig = process_human_model_output(
                                                                human_model_param=smplx_param, 
                                                                cam_param=self.cam_param, 
                                                                do_flip=do_flip, 
                                                                img_shape=img_shape, 
                                                                img2bb_trans=img2bb_trans,
                                                                rot=rot, 
                                                                human_model_type=model_type, 
                                                                joint_img=None if self.cam_param else joint_img,
                                                                body_3d_size=self.cfg.model.body_3d_size, 
                                                                hand_3d_size=self.cfg.model.hand_3d_size, 
                                                                face_3d_size=self.cfg.model.face_3d_size,
                                                                input_img_shape=self.cfg.model.input_img_shape, 
                                                                output_hm_shape=self.cfg.model.output_hm_shape, 
                                                                )

            lhand_valid = 1.0
            rhand_valid = 1.0
            # process the hand mesh for mano dataset
            if self.__class__.__name__ in ['FreiHand', 'InterHand', 'BlurHand', 'HanCo']:
                if (data['smplx_param']['lhand_root']==0).all(): 
                    lhand_valid = 0.0
                if (data['smplx_param']['rhand_root']==0).all():
                    rhand_valid = 0.0
                
                # build smplx but redo the hand rotation with global orientation

                smplx_pose_rotmat = batch_rodrigues(torch.Tensor(smplx_pose.reshape(-1,3))).reshape(smplx_pose.shape[0], -1)    

                # redo the hand oration: R_gt x R_inv x hand mesh
                R_gt_l = data['smplx_param']['lhand_root'] if 'lhand_root' in smplx_param else np.zeros((1, 3))
                R_gt_r = data['smplx_param']['rhand_root'] if 'rhand_root' in smplx_param else np.zeros((1, 3))

                R_gt_l = batch_rodrigues(torch.Tensor(R_gt_l.reshape(-1,3))).reshape(R_gt_l.shape[0], 3, 3) 
                R_gt_r = batch_rodrigues(torch.Tensor(R_gt_r.reshape(-1,3))).reshape(R_gt_r.shape[0], 3, 3)
                # import pdb; pdb.set_trace()

                # get hand mesh with wrong global orientation
                lhand_mesh = smplx_mesh_cam_orig[self.smpl_x.hand_vertex_idx['left_hand'], :]
                rhand_mesh = smplx_mesh_cam_orig[self.smpl_x.hand_vertex_idx['right_hand'], :]

                # get wrist offset and align hand mesh to pelvis
                lwrist_offset = np.dot(self.smpl_x.J_regressor, smplx_mesh_cam_orig)[self.smpl_x.J_regressor_idx['lwrist'], None, :]
                rwrist_offset = np.dot(self.smpl_x.J_regressor, smplx_mesh_cam_orig)[self.smpl_x.J_regressor_idx['rwrist'], None, :]
                mesh_out_lhand_align = lhand_mesh - lwrist_offset
                mesh_out_rhand_align = rhand_mesh - rwrist_offset

                # redo the rotation and align to wrist position world->cam
                R_gt_l = np.dot(data['extrinsic_r'], R_gt_l.squeeze())
                R_gt_r = np.dot(data['extrinsic_r'], R_gt_r.squeeze())

                mesh_global_lhand = np.dot(R_gt_l, mesh_out_lhand_align.T).T #+ lwrist_offset
                mesh_global_rhand = np.dot(R_gt_r, mesh_out_rhand_align.T).T #+ rwrist_offset
                
                # replace hand mesh in smplx mesh
                smplx_mesh_cam_orig[self.smpl_x.hand_vertex_idx['left_hand'], :] = mesh_global_lhand
                smplx_mesh_cam_orig[self.smpl_x.hand_vertex_idx['right_hand'], :] = mesh_global_rhand

            if self.__class__.__name__ in ['ARCTIC'] and (data['vertices3d'] != -1).all():
                smplx_mesh_cam_orig = data['vertices3d']

            data['joint_cam'][self.smpl_x.joint_part['lhand'], :] = (data['joint_cam'][self.smpl_x.joint_part['lhand'], :] - \
                data['joint_cam'][self.smpl_x.lwrist_idx, None,:]) * lhand_valid# left hand root-relative
            data['joint_cam'][self.smpl_x.joint_part['rhand'], :] = (data['joint_cam'][self.smpl_x.joint_part['rhand'], :] - \
                data['joint_cam'][self.smpl_x.rwrist_idx, None,:]) * rhand_valid

           
            inputs = {'img': img}
            targets = {'smplx_cam_trans' : smplx_cam_trans,
                    'smplx_mesh_cam': smplx_mesh_cam_orig,
                    'joint_cam': data['joint_cam'],}
            meta_info = {'bb2img_trans': bb2img_trans,
                        'gt_smplx_transl':smplx_cam_trans,
                        'lhand_valid': lhand_valid,
                        'rhand_valid': rhand_valid,
                        'focal': focal, 'principal_pt': princpt,
                        'img_id': data['idx']}

            return inputs, targets, meta_info

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans, input_img_shape, output_hm_shape):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / input_img_shape[1] * output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / input_img_shape[0] * output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0])
            xmax = np.max(bbox[:, 0])
            ymin = np.min(bbox[:, 1])
            ymax = np.max(bbox[:, 1])
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def evaluate(self, outs, cur_sample_idx=None):
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_l_hand': [], 'pa_mpvpe_r_hand': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 
                       'mpvpe_all': [], 'mpvpe_l_hand': [], 'mpvpe_r_hand': [], 'mpvpe_hand': [], 'mpvpe_face': [], 
                       'pa_mpjpe_body': [], 'pa_mpjpe_l_hand': [], 'pa_mpjpe_r_hand': [], 'pa_mpjpe_hand': [],
                       'mpjpe_body':[], 'mpjpe_l_hand': [], 'mpjpe_r_hand': [], 'mpjpe_hand': [],}


        for n in range(sample_num):
            out = outs[n]
            mesh_gt = out['smplx_mesh_cam_pseudo_gt']
            mesh_out = out['smplx_mesh_cam']


            if mesh_gt.shape[0] == 6890:
                face = self.smpl.face
                
                # root align -> ds (better for pve and mpjpe)
                mesh_out_root_align = mesh_out - np.dot(self.smpl_x.J_regressor, mesh_out)[self.smpl_x.J_regressor_idx['pelvis'], None,
                                            :] + np.dot(self.smpl.joint_regressor, mesh_gt)[self.smpl.orig_root_joint_idx, None,:]
                mesh_out_root_align = np.matmul(self.downsample_mat, mesh_out_root_align)

                # PVE from body
                mpvpe_all = np.sqrt(np.sum((mesh_out_root_align - mesh_gt) ** 2, 1)).mean() * 1000
                eval_result['mpvpe_all'].append(mpvpe_all)
                mesh_out_pa_align = rigid_align(mesh_out_root_align, mesh_gt)
                pa_mpvpe_all = np.sqrt(np.sum((mesh_out_pa_align - mesh_gt) ** 2, 1)).mean() * 1000
                eval_result['pa_mpvpe_all'].append(pa_mpvpe_all)
                
                # MPJPE from body joints
                joint_gt_body = np.dot(self.smpl.joint_regressor, mesh_gt)[LSP_MAPPIMG, :]
                joint_out_body_root_align = np.dot(self.smpl.joint_regressor, mesh_out_root_align)[LSP_MAPPIMG, :]
                joint_out_body_pa_align = rigid_align(joint_out_body_root_align, joint_gt_body)
                
                eval_result['mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_root_align - joint_gt_body) ** 2, 1)).mean() * 1000)
                eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_pa_align - joint_gt_body) ** 2, 1)).mean() * 1000)

            else:

                # MPVPE from all vertices
                mesh_out_align = mesh_out - np.dot(self.smpl_x.J_regressor, mesh_out)[self.smpl_x.J_regressor_idx['pelvis'], None,
                                            :] + np.dot(self.smpl_x.J_regressor, mesh_gt)[self.smpl_x.J_regressor_idx['pelvis'], None, :]
                joint_out_body_root_align = np.dot(self.smpl_x.j14_regressor, mesh_out_align)

                mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
                eval_result['mpvpe_all'].append(mpvpe_all)
                mesh_out_align = rigid_align(mesh_out, mesh_gt)
                pa_mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
                eval_result['pa_mpvpe_all'].append(pa_mpvpe_all)


                mesh_gt_lhand = mesh_gt[self.smpl_x.hand_vertex_idx['left_hand'], :] - np.dot(
                    self.smpl_x.J_regressor, mesh_gt)[self.smpl_x.J_regressor_idx['lwrist'], None, :]
                mesh_gt_rhand = mesh_gt[self.smpl_x.hand_vertex_idx['right_hand'], :] - np.dot(
                    self.smpl_x.J_regressor, mesh_gt)[self.smpl_x.J_regressor_idx['rwrist'], None, :]

                mesh_out_lhand = mesh_out[self.smpl_x.hand_vertex_idx['left_hand'], :]
                mesh_out_rhand = mesh_out[self.smpl_x.hand_vertex_idx['right_hand'], :]
                mesh_out_lhand_align = mesh_out_lhand - np.dot(self.smpl_x.J_regressor, mesh_out)[
                                                        self.smpl_x.J_regressor_idx['lwrist'], None, :] 
                mesh_out_rhand_align = mesh_out_rhand - np.dot(self.smpl_x.J_regressor, mesh_out)[
                                                        self.smpl_x.J_regressor_idx['rwrist'], None, :] 
                
                if out['lhand_valid']:
                    eval_result['mpvpe_l_hand'].append(np.sqrt(
                        np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
                if out['rhand_valid']:
                    eval_result['mpvpe_r_hand'].append(np.sqrt(
                        np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
                hand_mpve_all = (np.sqrt(
                    np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 * out['lhand_valid'] + np.sqrt(
                    np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000 * out['rhand_valid']
                    ) / (out['lhand_valid'] + out['rhand_valid'])

                eval_result['mpvpe_hand'].append(hand_mpve_all)

                mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
                mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)

                if out['lhand_valid']:
                    eval_result['pa_mpvpe_l_hand'].append(np.sqrt(
                        np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
                if out['rhand_valid']:      
                    eval_result['pa_mpvpe_r_hand'].append(np.sqrt(
                        np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)

                eval_result['pa_mpvpe_hand'].append((np.sqrt(
                    np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 * out['lhand_valid'] + np.sqrt(
                    np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000 * out['rhand_valid']) / 
                    (out['lhand_valid'] + out['rhand_valid']))

                # MPVPE from face vertices
                mesh_gt_face = mesh_gt[self.smpl_x.face_vertex_idx, :]
                mesh_out_face = mesh_out[self.smpl_x.face_vertex_idx, :]
                mesh_out_face_align = mesh_out_face - np.dot(self.smpl_x.J_regressor, mesh_out)[self.smpl_x.J_regressor_idx['neck'],
                                                    None, :] + np.dot(self.smpl_x.J_regressor, mesh_gt)[
                                                                self.smpl_x.J_regressor_idx['neck'], None, :]
                eval_result['mpvpe_face'].append(
                    np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)
                mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
                eval_result['pa_mpvpe_face'].append(
                    np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)

                joint_gt_body = np.dot(self.smpl_x.j14_regressor, mesh_gt)
                joint_out_body = np.dot(self.smpl_x.j14_regressor, mesh_out)
                joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
                
                eval_result['mpjpe_body'].append(
                    np.sqrt(np.sum((joint_out_body_root_align - joint_gt_body) ** 2, 1)).mean() * 1000)
                eval_result['pa_mpjpe_body'].append(
                    np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)).mean() * 1000)

                joint_gt_lhand = np.dot(self.smpl_x.orig_hand_regressor['left'], mesh_gt)[1:]
                joint_gt_rhand = np.dot(self.smpl_x.orig_hand_regressor['right'], mesh_gt)[1:]
                

                joint_out_lhand = np.dot(self.smpl_x.orig_hand_regressor['left'], mesh_out)[1:] - np.dot(self.smpl_x.J_regressor, mesh_out)[
                                                        self.smpl_x.J_regressor_idx['lwrist'], None, :]
                
                joint_out_rhand = np.dot(self.smpl_x.orig_hand_regressor['right'], mesh_out)[1:] - np.dot(self.smpl_x.J_regressor, mesh_out)[
                                                        self.smpl_x.J_regressor_idx['rwrist'], None, :]


                joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
                joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)
                
                if out['lhand_valid']:
                    eval_result['mpjpe_l_hand'].append(np.sqrt(
                        np.sum((joint_out_lhand - joint_gt_lhand) ** 2, 1)).mean() * 1000)
                if out['rhand_valid']:
                    eval_result['mpjpe_r_hand'].append(np.sqrt(
                        np.sum((joint_out_rhand - joint_gt_rhand) ** 2, 1)).mean() * 1000)
                
                hand_pa_mpve_all = (np.sqrt(
                    np.sum((joint_out_lhand - joint_gt_lhand) ** 2, 1)).mean() * 1000 * out['lhand_valid'] + np.sqrt(
                    np.sum((joint_out_rhand - joint_gt_rhand) ** 2, 1)).mean() * 1000 * out['rhand_valid']
                    ) / (out['lhand_valid'] + out['rhand_valid'])

                eval_result['mpjpe_hand'].append(hand_pa_mpve_all)
                
                if out['lhand_valid']:
                    value = np.sqrt(np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000
                    
                    if value < 100:
                        eval_result['pa_mpjpe_l_hand'].append(value)
                    if value > 100:
                        print("lhand:",value)
                        continue

                if out['rhand_valid']:
                    value = np.sqrt(np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000
                    
                    if value < 100:
                        eval_result['pa_mpjpe_r_hand'].append(value)
                    if value > 100:
                        print("rhand:",value)
                        continue
                    
                eval_result['pa_mpjpe_hand'].append((np.sqrt(
                    np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000 * out['lhand_valid'] + np.sqrt(
                    np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000 * out['rhand_valid']
                    ) / (out['lhand_valid'] + out['rhand_valid']))
                

        return eval_result
            
    def print_eval_result(self, eval_result):
        print(f'======{self.cfg.data.testset}======')
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
        print('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))
        print()

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
        print('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
        print()
        
        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        print('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
        print('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
        print('PA MPJPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand']))
        print()

        print('MPJPE (Body): %.2f mm' % np.mean(eval_result['mpjpe_body']))
        print('MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['mpjpe_l_hand']))
        print('MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['mpjpe_r_hand']))
        print('MPJPE (Hands): %.2f mm' % np.mean(eval_result['mpjpe_hand']))
        print()

        print(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])},"
        f"{np.mean(eval_result['pa_mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_l_hand'])},{np.mean(eval_result['pa_mpjpe_r_hand'])},{np.mean(eval_result['pa_mpjpe_hand'])}")
        print()


        f = open(os.path.join(self.cfg.log.result_dir, 'result.txt'), 'w')
        f.write(f'{self.cfg.data.testset} dataset \n')
        f.write('PA MPVPE (All): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_all']))
        f.write('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
        f.write('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
        f.write('PA MPVPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_hand']))
        f.write('PA MPVPE (Face): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_face']))
        f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
        f.write('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
        f.write('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
        f.write('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        f.write('MPVPE (Face): %.2f mm\n' % np.mean(eval_result['mpvpe_face']))
        f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))
        f.write('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
        f.write('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
        f.write('PA MPJPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_hand']))
        f.write(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])},"
        f"{np.mean(eval_result['pa_mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_l_hand'])},{np.mean(eval_result['pa_mpjpe_r_hand'])},{np.mean(eval_result['pa_mpjpe_hand'])}")
        f.close()

    def decompress_keypoints(self, humandata) -> None:
        """If a key contains 'keypoints', and f'{key}_mask' is in self.keys(),
        invalid zeros will be inserted to the right places and f'{key}_mask'
        will be unlocked.

        Raises:
            KeyError:
                A key contains 'keypoints' has been found
                but its corresponding mask is missing.
        """
        assert bool(humandata['__keypoints_compressed__']) is True
        key_pairs = []
        for key in humandata.files:
            if key not in KPS2D_KEYS + KPS3D_KEYS:
                continue
            mask_key = f'{key}_mask'
            if mask_key in humandata.files:
                print(f'Decompress {key}...')
                key_pairs.append([key, mask_key])
        decompressed_dict = {}
        for kpt_key, mask_key in key_pairs:
            mask_array = np.asarray(humandata[mask_key])
            compressed_kpt = humandata[kpt_key]
            kpt_array = \
                self.add_zero_pad(compressed_kpt, mask_array)
            decompressed_dict[kpt_key] = kpt_array
        del humandata
        return decompressed_dict

    def add_zero_pad(self, compressed_array: np.ndarray,
                         mask_array: np.ndarray) -> np.ndarray:
        """Pad zeros to a compressed keypoints array.

        Args:
            compressed_array (np.ndarray):
                A compressed keypoints array.
            mask_array (np.ndarray):
                The mask records compression relationship.

        Returns:
            np.ndarray:
                A keypoints array in full-size.
        """
        if compressed_array.shape[1] == mask_array.shape[0]:
            print("No need to decompress")
            return compressed_array
        else:
            assert mask_array.sum() == compressed_array.shape[1]
            data_len, _, dim = compressed_array.shape
            mask_len = mask_array.shape[0]
            ret_value = np.zeros(
                shape=[data_len, mask_len, dim], dtype=compressed_array.dtype)
            valid_mask_index = np.where(mask_array == 1)[0]
            ret_value[:, valid_mask_index, :] = compressed_array
            return ret_value
