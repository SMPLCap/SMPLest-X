import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import math
import copy
from models.module import TransformerDecoderHead, ViT
from models.loss import CoordLoss, ParamLoss
from human_models.human_models import SMPL, SMPLX
from utils.transforms import rot6d_to_axis_angle, batch_rodrigues, rot6d_to_rotmat
from utils.data_utils import load_img


class Model(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(Model, self).__init__()
        self.smpl_x = SMPLX.get_instance()
        
        # network
        self.cfg = config
        self.encoder = encoder
        self.decoder = decoder

        # loss
        self.smplx_layer = copy.deepcopy(self.smpl_x.layer['neutral']).cuda()
        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

        self.body_num_joints = len(self.smpl_x.pos_joint_part['body'])
        self.hand_joint_num = len(self.smpl_x.pos_joint_part['rhand'])

        # num of parameters
        self.trainable_modules = [self.encoder, self.decoder]
        param_net = 0
        for module in self.trainable_modules:
            param_net += sum(p.numel() for p in module.parameters() if p.requires_grad)

        print(f'Total #parameters: {param_net} ({param_net/1000000000:.2f}B)')


    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.cfg.model.focal[0] * self.cfg.model.focal[1] * 
                            self.cfg.model.camera_3d_size * self.cfg.model.camera_3d_size / (
                self.cfg.model.input_body_shape[0] * self.cfg.model.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        # transl=cam_trans, 
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  transl=cam_trans, left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        if mode == 'test' and self.cfg.data.testset in ['AGORA_test', 'BEDLAM_test']:  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, self.smpl_x.joint_idx, :]

        # project 3D coordinates to 2D space
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * \
            self.cfg.model.focal[0] + self.cfg.model.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * \
            self.cfg.model.focal[1] + self.cfg.model.princpt[1]
        x = x / self.cfg.model.input_body_shape[1] * self.cfg.model.output_hm_shape[2]
        y = y / self.cfg.model.input_body_shape[0] * self.cfg.model.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, self.smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        # mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering
        joint_cam_wo_ra = joint_cam.clone()

        # left hand root (left wrist)-relative 3D coordinatese
        lhand_idx = self.smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, self.smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinatese
        rhand_idx = self.smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, self.smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = self.smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, self.smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam, root_cam

    def generate_mesh_gt(self, targets, mode):
        if 'smplx_mesh_cam' in targets:
            return targets['smplx_mesh_cam'], None
        nums = [3, 63, 45, 45, 3]
        accu = []
        temp = 0
        for num in nums:
            temp += num
            accu.append(temp)
        pose = targets['smplx_pose']
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose = \
            pose[:, :accu[0]], pose[:, accu[0]:accu[1]], pose[:, accu[1]:accu[2]], pose[:, accu[2]:accu[3]], pose[:,
                                                                                                             accu[3]:
                                                                                                             accu[4]]
        shape = targets['smplx_shape']
        expr = targets['smplx_expr']
        cam_trans = targets['smplx_cam_trans']

        # final output
        joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam, root_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
                                                         expr, cam_trans, mode)

        return mesh_cam, root_cam

    def batch_hand_global_rotation(self, rot_mat, hand_idx):
        """
        rot_mat: [b, 22, 3, 3]
        """
        hand_global_rotmat = []
        for item in rot_mat:
            parents = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                16, 17, 18, 19], dtype=torch.int64)
            transforms_mat = item.clone()
            transform_chain = [transforms_mat[0].detach()] # pelvis
            
            for i in range(1, parents.shape[0]):
                # Subtract the joint location at the rest pose
                # No need for rotation, since it's identity when at rest
                curr_res = torch.matmul(transform_chain[parents[i]],
                                        transforms_mat[i])
                transform_chain.append(curr_res)
            transforms = torch.stack(transform_chain, dim=0)
            hand_global_rotmat.append(transforms[hand_idx].unsqueeze(0))

        batch_hand_global_rotmat = torch.cat(hand_global_rotmat, dim=0)
        return batch_hand_global_rotmat

    def forward(self, inputs, targets, meta_info, mode):
        body_img = F.interpolate(inputs['img'], self.cfg.model.input_body_shape)    #READ: is this image the bb image

        # 1. Encoder (frozen — no gradients needed)
        with torch.no_grad():
            img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]

        # 2. Decoder
        pred_mano_params = self.decoder(task_tokens, img_feat)

        # get transl
        body_trans = self.get_camera_trans(pred_mano_params['body_cam'])
        lhand_trans = self.get_camera_trans(pred_mano_params['lhand_cam'])
        rhand_trans = self.get_camera_trans(pred_mano_params['rhand_cam'])
        face_trans = self.get_camera_trans(pred_mano_params['face_cam'])

        # convert predicted rot6d to aa (not unique convention may cause problem)
        root_pose_aa = rot6d_to_axis_angle(pred_mano_params['body_root_pose'])
        body_pose_aa = rot6d_to_axis_angle(pred_mano_params['body_pose'].reshape(-1, 6)).reshape(pred_mano_params['body_pose'].shape[0], -1) 
        
        lhand_root_pose_aa = rot6d_to_axis_angle(pred_mano_params['lhand_root_pose'])
        rhand_root_pose_aa = rot6d_to_axis_angle(pred_mano_params['rhand_root_pose'])
        lhand_pose_aa= rot6d_to_axis_angle(pred_mano_params['lhand_pose'].reshape(-1, 6)).reshape(pred_mano_params['lhand_pose'].shape[0], -1)  
        rhand_pose_aa= rot6d_to_axis_angle(pred_mano_params['rhand_pose'].reshape(-1, 6)).reshape(pred_mano_params['rhand_pose'].shape[0], -1)
        
        face_root_pose = rot6d_to_axis_angle(pred_mano_params['face_root_pose'])
        face_jaw_pose_aa = rot6d_to_axis_angle(pred_mano_params['face_jaw_pose'])
        
        # convert predicted aa to rotmat 
        root_pose_rotmat = batch_rodrigues(root_pose_aa.reshape(-1, 3)).reshape(root_pose_aa.shape[0], -1)
        body_pose_rotmat = batch_rodrigues(body_pose_aa.reshape(-1, 3)).reshape(body_pose_aa.shape[0], -1)  
        full_body_rotmat = torch.cat((root_pose_rotmat, body_pose_rotmat), 1)

        lhand_root_pose_rotmat = batch_rodrigues(lhand_root_pose_aa.reshape(-1, 3)).reshape(lhand_root_pose_aa.shape[0], -1)
        rhand_root_pose_rotmat = batch_rodrigues(rhand_root_pose_aa.reshape(-1, 3)).reshape(rhand_root_pose_aa.shape[0], -1)
        lhand_pose_rotmat = batch_rodrigues(lhand_pose_aa.reshape(-1, 3)).reshape(lhand_pose_aa.shape[0], -1) 
        rhand_pose_rotmat = batch_rodrigues(rhand_pose_aa.reshape(-1, 3)).reshape(rhand_pose_aa.shape[0], -1)  

        face_root_rotmat = batch_rodrigues(face_root_pose.reshape(-1, 3)).reshape(face_root_pose.shape[0], -1)
        face_jaw_pose_rotmat = batch_rodrigues(face_jaw_pose_aa.reshape(-1, 3)).reshape(face_jaw_pose_aa.shape[0], -1)  

        pose = torch.cat((root_pose_rotmat, body_pose_rotmat, lhand_pose_rotmat, rhand_pose_rotmat, face_jaw_pose_rotmat), 1)

        # kps2d, kps3d and mesh output
        joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam, pred_root_cam = self.get_coord(root_pose_aa, body_pose_aa, lhand_pose_aa, 
                                                                            rhand_pose_aa, face_jaw_pose_aa, 
                                                                            pred_mano_params['body_betas'], 
                                                                            pred_mano_params['face_expression'], 
                                                                            body_trans, mode)
        
        if mode == 'train':
            loss = {}

            smplx_kps_3d_weight = getattr(self.cfg.train, 'smplx_kps_3d_weight', 1.0)
            smplx_kps_2d_weight = getattr(self.cfg.train, 'smplx_kps_2d_weight', 1.0)
            smplx_pose_weight = getattr(self.cfg.train, 'smplx_pose_weight', 1.0)
            smplx_shape_weight = getattr(self.cfg.train, 'smplx_shape_weight', 1.0)
            smplx_orient_weight = getattr(self.cfg.train, 'smplx_orient_weight', 1.0)
            smplx_hand_kps_3d_weight = getattr(self.cfg.train, 'smplx_hand_kps_3d_weight', 1.0)
            hand_root_weight = getattr(self.cfg.train, 'hand_root_weight', 1.0)

            ### pose loss ###
            targets['smplx_pose_rotmat'] = batch_rodrigues(targets['smplx_pose'].reshape(-1,3)).reshape(
                                                                    targets['smplx_pose'].shape[0], -1) 
           
            # do not supervise root pose if original agora json is used
            loss['smplx_orient'] = self.param_loss(pose, targets['smplx_pose_rotmat'], 
                                                meta_info['smplx_pose_valid'])[:, :9] * smplx_orient_weight
            loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose_rotmat'], 
                                                meta_info['smplx_pose_valid']) * smplx_pose_weight
            ### shape loss ###
            loss['smplx_shape'] = self.param_loss(pred_mano_params['body_betas'], targets['smplx_shape'],
                                                  meta_info['smplx_shape_valid'][:, None]) * smplx_shape_weight 
            ### expression loss ###
            loss['smplx_expr'] = self.param_loss(pred_mano_params['face_expression'], targets['smplx_expr'], 
                                                meta_info['smplx_expr_valid'][:, None])

            ### keypoints3d wo/ ra loss ###
            meta_info['root_valid'] = meta_info['smplx_pose_valid'][:, 0] != 0 
            
            hand_index = list(self.smpl_x.joint_part['lhand']) + list(self.smpl_x.joint_part['rhand'])
            
            #---LOSSES MAPPING to humandata
            #joint_cam = keypoints_3d
            #joint_proj = keypoints_2d
            #smplx_joint_cam = 3d keypoints obtained from forward passing smpl parameters, for our case these are the same


            # if root orientation not given, ignore loss wo/ ra, only for full-body dataset
            loss['joint_cam'] = self.coord_loss(joint_cam_wo_ra, targets['joint_cam'], 
                                    meta_info['joint_trunc'] * meta_info['is_3D'][:, None, None]* meta_info['joint_trunc'][:,0, :][:, None]) * smplx_kps_3d_weight

            if getattr(self.cfg.train, 'hand_loss', False):
                ### 3d hand kps loss wo/ ra for hand alignment, only for full-body datset
                loss['hand_align'] = self.coord_loss(joint_cam_wo_ra[:, hand_index, :], targets['joint_cam'][:, hand_index, :], 
                                    meta_info['joint_trunc'][:, hand_index, :] * meta_info['is_3D'][:, None, None]* meta_info['joint_trunc'][:,0, :][:, None]) * smplx_hand_kps_3d_weight

            ### keypoints3d w/ ra loss ###
            loss['smplx_joint_cam'] = self.coord_loss(joint_cam, targets['smplx_joint_cam'], 
                                    meta_info['joint_trunc']) * smplx_kps_3d_weight

            if getattr(self.cfg.train, 'hand_loss', False):
                ### 3d hand kps loss w/ ra for hand pose
                loss['hand_pose'] = self.coord_loss(joint_cam[:, hand_index, :], targets['smplx_joint_cam'][:, hand_index, :], 
                                   meta_info['joint_trunc'][:, hand_index, :]) * smplx_hand_kps_3d_weight

            ### keypoints2d loss ###
            loss['joint_proj'] = self.coord_loss(joint_proj[..., :2], targets['joint_img'][:, :, :2], 
                                            meta_info['joint_trunc']) * smplx_kps_2d_weight
            
            ### hand and face consistency loss, part global orientation loss ###
            # meta_info['root_valid'] = meta_info['smplx_pose_valid'][:, 0] != 0
            # hand pose validity
            lwrist_pose_idx = self.smpl_x.orig_joints_name.index('L_Wrist')
            rwrist_pose_idx = self.smpl_x.orig_joints_name.index('R_Wrist')
            lhand_thumb_id = self.smpl_x.orig_joints_name.index('L_Thumb_1')
            rhand_thumb_id = self.smpl_x.orig_joints_name.index('R_Thumb_1')

            # pred_pelvis -> pred_hand global orientation
            #opt: Skip all hand trnsformations/loss calculations
            if hand_root_weight > 0:
                lhand_root_rotmat_chain = self.batch_hand_global_rotation(full_body_rotmat.view(full_body_rotmat.shape[0], -1, 3, 3),
                                                                    lwrist_pose_idx).view(-1, 9)
                rhand_root_rotmat_chain = self.batch_hand_global_rotation(full_body_rotmat.view(full_body_rotmat.shape[0], -1, 3, 3),
                                                                    rwrist_pose_idx).view(-1, 9)

                # -->full body dataset <--
                # gt_pelvis -> gt_hand global orientation, for full-body dataset
                lhand_root_pose_rotmat_chain_gt = self.batch_hand_global_rotation(targets['smplx_pose_rotmat'].view(targets['smplx_pose_rotmat'].shape[0], -1, 3, 3)[:, :22, ...],
                                                                    lwrist_pose_idx).view(-1, 9)
                rhand_root_pose_rotmat_chain_gt = self.batch_hand_global_rotation(targets['smplx_pose_rotmat'].view(targets['smplx_pose_rotmat'].shape[0], -1, 3, 3)[:, :22, ...],
                                                                    rwrist_pose_idx).view(-1, 9)
            
            if hand_root_weight > 0:
                lhand_valid = meta_info['smplx_pose_valid'].view(meta_info['smplx_pose_valid'].shape[0], -1, 9)[:, lwrist_pose_idx]
                rhand_valid = meta_info['smplx_pose_valid'].view(meta_info['smplx_pose_valid'].shape[0], -1, 9)[:, rwrist_pose_idx]

                lhand_root_loss = self.param_loss(lhand_root_pose_rotmat_chain_gt, lhand_root_pose_rotmat, lhand_valid) * meta_info['joint_trunc'][:,0, :][:, None]
                rhand_root_loss = self.param_loss(rhand_root_pose_rotmat_chain_gt, rhand_root_pose_rotmat, rhand_valid) * meta_info['joint_trunc'][:,0, :][:, None]
                loss['hand_root'] = (lhand_root_loss + rhand_root_loss) * hand_root_weight

                if not getattr(self.cfg.train, 'no_chain_hand_loss', False):
                    lhand_root_loss_chain = self.param_loss(lhand_root_pose_rotmat_chain_gt, lhand_root_rotmat_chain, lhand_valid) * meta_info['joint_trunc'][:,0, :][:, None]
                    rhand_root_loss_chain = self.param_loss(rhand_root_pose_rotmat_chain_gt, rhand_root_rotmat_chain, rhand_valid) * meta_info['joint_trunc'][:,0, :][:, None]
                    loss['hand_root_chain'] = (lhand_root_loss_chain + rhand_root_loss_chain) * hand_root_weight

                # --> hand only dataset <--
                lhand_valid = meta_info['smplx_pose_valid'].view(meta_info['smplx_pose_valid'].shape[0], -1, 9)[:, lhand_thumb_id]
                rhand_valid = meta_info['smplx_pose_valid'].view(meta_info['smplx_pose_valid'].shape[0], -1, 9)[:, rhand_thumb_id]

                targets['lhand_root_rotmat'] = batch_rodrigues(targets['lhand_root'].reshape(-1,3)).reshape(targets['lhand_root'].shape[0], -1)
                targets['rhand_root_rotmat'] = batch_rodrigues(targets['rhand_root'].reshape(-1,3)).reshape(targets['rhand_root'].shape[0], -1)
                lhand_root_loss = self.param_loss(targets['lhand_root_rotmat'], lhand_root_pose_rotmat, lhand_valid) * (1 - meta_info['joint_trunc'][:,0, :][:, None])
                rhand_root_loss = self.param_loss(targets['rhand_root_rotmat'], rhand_root_pose_rotmat, rhand_valid) * (1 - meta_info['joint_trunc'][:,0, :][:, None])
                loss['hand_root'] += (lhand_root_loss + rhand_root_loss) * hand_root_weight

                if not getattr(self.cfg.train, 'no_chain_hand_loss', False):
                    lhand_root_loss_chain = self.param_loss(targets['lhand_root_rotmat'], lhand_root_rotmat_chain, lhand_valid) * (1 - meta_info['joint_trunc'][:,0, :][:, None])
                    rhand_root_loss_chain = self.param_loss(targets['rhand_root_rotmat'], rhand_root_rotmat_chain, rhand_valid) * (1 - meta_info['joint_trunc'][:,0, :][:, None])
                    loss['hand_root_chain'] += (lhand_root_loss_chain + rhand_root_loss_chain) * hand_root_weight
            else:
                loss['hand_root'] = torch.zeros(1, device=body_img.device)
                if not getattr(self.cfg.train, 'no_chain_hand_loss', False):
                    loss['hand_root_chain'] = torch.zeros(1, device=body_img.device)
        
            return loss

        else:
            if mode == 'test' and 'smplx_mesh_cam' in targets:
                mesh_pseudo_gt, _ = self.generate_mesh_gt(targets, mode)

            # test output
            out = {}
            out['img'] = inputs['img']
            out['smplx_joint_proj'] = joint_proj
            out['smplx_mesh_cam'] = mesh_cam
            out['smplx_root_pose'] = root_pose_aa
            out['smplx_body_pose'] = body_pose_aa
            out['smplx_lhand_pose'] = lhand_pose_aa
            out['smplx_rhand_pose'] = rhand_pose_aa
            out['smplx_jaw_pose'] = face_jaw_pose_aa
            out['smplx_shape'] = pred_mano_params['body_betas']
            out['smplx_expr'] = pred_mano_params['face_expression']
            out['cam_trans'] = body_trans
            out['smplx_joint_cam'] = joint_cam_wo_ra # for bedlam test


            if 'smplx_shape' in targets:
                out['smplx_shape_target'] = targets['smplx_shape']
            if 'img_path' in meta_info:
                out['img_path'] = meta_info['img_path']
            if 'smplx_mesh_cam' in targets:
                out['smplx_mesh_cam_pseudo_gt'] = mesh_pseudo_gt
            if 'smplx_mesh_cam' in targets:
                out['smplx_mesh_cam_target'] = targets['smplx_mesh_cam']
            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'gt_smplx_transl' in meta_info:
                out['gt_smplx_transl'] = meta_info['gt_smplx_transl']
            if 'joint_cam' in targets:
                out['joint_cam'] = targets['joint_cam']
            if 'lhand_valid' in meta_info:
                out['lhand_valid'] = meta_info['lhand_valid']
            if 'rhand_valid' in meta_info:
                out['rhand_valid'] = meta_info['rhand_valid']
            if 'img_id' in meta_info:
                out['img_id'] = meta_info['img_id']

            return out
    

def get_model(cfg, mode):

    encoder = ViT(**cfg.model.encoder_config)
    if mode == 'train':
        encoder_pretrained_model = torch.load(cfg.model.encoder_pretrained_model_path)['state_dict']
        encoder.load_state_dict(encoder_pretrained_model, strict=False)
        print(f"Initialized encoder from {cfg.model.encoder_pretrained_model_path}")

    decoder = TransformerDecoderHead(**cfg.model.decoder_config) 

    model = Model(cfg, encoder, decoder)
    return model