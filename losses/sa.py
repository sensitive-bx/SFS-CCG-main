# """Custom losses."""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# __all__ = ['CriterionSA']
#
# class CriterionSA(nn.Module):  # CKA between self-attention modules
#
#     def __init__(self, temperature=1.0, temperaturesa=1.0):
#
#         super(CriterionSA, self).__init__()
#         self.temperature = temperature # Temperature of score map
#         self.temperaturesa=temperaturesa # Temperature of feature map
#         self.softmax = torch.nn.Softmax(dim=-1)
#         self.gammacam = nn.Parameter(torch.zeros(1))
#         self.gammapam = nn.Parameter(torch.zeros(1))
#         self.kld = nn.KLDivLoss(reduction='mean')
#         self.mse = nn.MSELoss(reduction='mean')
#
#     def centering(self, K):
#         n = K.shape[0]
#         unit = torch.ones([n, n]).cuda()
#         I = torch.eye(n).cuda()
#         H = I - unit / n
#         # H.cuda()
#         return torch.matmul(torch.matmul(H, K), H)
#
#     def linear_HSIC(self, X, Y):
#         L_X = torch.matmul(X, X.T)
#         L_Y = torch.matmul(Y, Y.T)
#         return torch.sum(self.centering(L_X) * self.centering(L_Y))
#
#     def linear_CKA_loss(self, X, Y): # CKA between X and Y
#         hsic = self.linear_HSIC(X, Y)
#         var1 = torch.sqrt(self.linear_HSIC(X, X))
#         var2 = torch.sqrt(self.linear_HSIC(Y, Y))
#         return -torch.log(torch.mean(torch.abs(torch.div(hsic, (var1 * var2)))) + 1e-8)
#
#     def CAM(self, X): # Channel Attention Module
#         m_batchsize, C, height, width = X.size()
#         proj_query = X.contiguous().view(m_batchsize, C, -1) # reshape
#         proj_key = X.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1) # reshape and transpose
#         energy = torch.bmm(proj_query, proj_key) # multiplication
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         attention = self.softmax(energy_new/self.temperaturesa)
#         proj_value = X.contiguous().view(m_batchsize, C, -1)
#         out = torch.bmm(attention, proj_value)
#         out = out.contiguous().view(m_batchsize, C, height, width)
#         out = self.gammacam*out + X
#         return out
#
#     def PAM(self, X): # Positionnal Attention Module
#         m_batchsize, C, height, width = X.size()
#         in_dim = C
#         self.query_conv =  torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1).cuda() # to generate B
#         self.key_conv =  torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1).cuda() # to generate C
#         self.value_conv =  torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).cuda()
#         proj_query = self.query_conv(X).contiguous().view(m_batchsize, -1, width*height).permute(0, 2, 1) # B
#         proj_key = self.key_conv(X).contiguous().view(m_batchsize, -1, width*height) # C
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy/self.temperaturesa) # S
#         proj_value = self.value_conv(X).contiguous().view(m_batchsize, -1, width*height)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.contiguous().view(m_batchsize, C, height, width)
#         out = self.gammapam*out + X
#         return out
#
#     def forward(self,feat_S, feat_T, feature_transform="CAM"):  # CKA between CAM and between PAM
#         # feat_S, feat_T = feat_S.cuda(), feat_T.cuda()
#         m_batchsize, CS, height, width = feat_S.size()
#         m_batchsize, CT, height, width = feat_T.size()
#         loss_CAM, loss_PAM = 0, 0
#         self.conv = nn.Conv2d(CS, CT, kernel_size=1, bias=False).cuda()
#
#         if feature_transform=="CAM_MSE":
#             feat_S = self.conv(feat_S)
#             CAM_S = self.CAM(feat_S)
#             CAM_T = self.CAM(feat_T)
#             loss_CAM = self.mse(CAM_S, CAM_T)
#             return loss_CAM
#         elif feature_transform=="CAM_CKA":
#             CAM_S = self.CAM(feat_S)
#             CAM_T = self.CAM(feat_T)
#             CAM_S = CAM_S.view(CAM_S.size(0), -1)
#             CAM_T = CAM_T.view(CAM_T.size(0), -1)
#             loss_CAM = self.linear_CKA_loss(CAM_S, CAM_T)
#             return loss_CAM
#         elif feature_transform=="PAM_CKA":
#             PAM_S = self.PAM(feat_S)
#             PAM_T = self.PAM(feat_T)
#             PAM_S = PAM_S.view(PAM_S.size(0), -1)
#             PAM_T = PAM_T.view(PAM_T.size(0), -1)
#             loss_PAM = self.linear_CKA_loss(PAM_S, PAM_T)
#             return loss_PAM
#         elif feature_transform=="gridPAM_MSE":
#             # Student feature maps to blocks
#             feat_S = self.conv(feat_S)
#             firstpart_S = torch.chunk(feat_S, 5, dim=2)
#             partsPAM_S = []
#             for i in firstpart_S:
#                 secondpart_S = torch.chunk(i, 5, dim=3)
#                 for j in secondpart_S:
#                     partsPAM_S.append(self.PAM(j))
#             # Teacher feature maps to blocks
#             firstpart_T = torch.chunk(feat_T, 5, dim=2)
#             partsPAM_T = []
#             for i in firstpart_T:
#                 secondpart_T = torch.chunk(i, 5, dim=3)
#                 for j in secondpart_T:
#                     partsPAM_T.append(self.PAM(j))
#             # Loss computation
#             n = len(partsPAM_S)
#             loss_PAM = 0
#             for i in range(0, n):
#                 loss_PAM += self.mse(partsPAM_S[i], partsPAM_T[i])
#             loss = loss_PAM/n
#             return loss
#         elif feature_transform == "gridPAM_CKA":
#             # Student feature maps to blocks
#             firstpart_S = torch.chunk(feat_S, 5, dim=2)
#             partsPAM_S = []
#             for i in firstpart_S:
#                 secondpart_S = torch.chunk(i, 5, dim=3)
#                 for j in secondpart_S:
#                     partsPAM_S.append(self.PAM(j))
#             # Teacher feature maps to blocks
#             firstpart_T = torch.chunk(feat_T, 5, dim=2)
#             partsPAM_T = []
#             for i in firstpart_T:
#                 secondpart_T = torch.chunk(i, 5, dim=3)
#                 for j in secondpart_T:
#                     partsPAM_T.append(self.PAM(j))
#             # Loss computation
#             n = len(partsPAM_S)
#             loss_PAM = 0
#             for i in range(0, n):
#                 # -------- CKA gridCAM
#                 S_cam = partsPAM_S[i].view(partsPAM_S[i].size(0), -1)
#                 T_cam = partsPAM_T[i].view(partsPAM_T[i].size(0), -1)
#                 loss_PAM += self.linear_CKA_loss(S_cam, T_cam)
#             loss = loss_PAM/n
#             return loss
#         elif feature_transform=="separately_CAMgridPAM_MSE":
#             feat_S = self.conv(feat_S)
#             # ---- gridPAM
#             # Student feature maps to blocks
#             firstpart_S = torch.chunk(feat_S, 5, dim=2)
#             partsPAM_S = []
#             for i in firstpart_S:
#                 secondpart_S = torch.chunk(i, 5, dim=3)
#                 for j in secondpart_S:
#                     partsPAM_S.append(self.PAM(j))
#             # Teacher feature maps to blocks
#             firstpart_T = torch.chunk(feat_T, 5, dim=2)
#             partsPAM_T = []
#             for i in firstpart_T:
#                 secondpart_T = torch.chunk(i, 5, dim=3)
#                 for j in secondpart_T:
#                     partsPAM_T.append(self.PAM(j))
#             # Loss computation
#             n = len(partsPAM_S)
#             loss_PAM = 0
#             for i in range(0, n):
#                 loss_PAM += self.mse( partsPAM_S[i], partsPAM_T[i])
#             loss_PAM = loss_PAM/n
#             # ---- CAM
#             CAM_S = self.CAM(feat_S)
#             CAM_T = self.CAM(feat_T)
#             loss_CAM = self.mse(CAM_S, CAM_T)
#             return loss_CAM, loss_PAM
#         elif feature_transform=="separately_CAMgridPAM_CKA":
#             # ---- gridPAM
#             # Student feature maps to blocks
#             firstpart_S = torch.chunk(feat_S, 5, dim=2)
#             partsPAM_S = []
#             for i in firstpart_S:
#                 secondpart_S = torch.chunk(i, 5, dim=3)
#                 for j in secondpart_S:
#                     partsPAM_S.append(self.PAM(j))
#             # Teacher feature maps to blocks
#             firstpart_T = torch.chunk(feat_T, 5, dim=2)
#             partsPAM_T = []
#             for i in firstpart_T:
#                 secondpart_T = torch.chunk(i, 5, dim=3)
#                 for j in secondpart_T:
#                     partsPAM_T.append(self.PAM(j))
#             # Loss computation
#             n = len(partsPAM_S)
#             loss_PAM = 0
#             for i in range(0, n):
#                 S_pam = partsPAM_S[i].view(partsPAM_S[i].size(0), -1)
#                 T_pam = partsPAM_T[i].view(partsPAM_T[i].size(0), -1)
#                 loss_PAM += self.linear_CKA_loss(S_pam, T_pam)
#             loss_PAM = loss_PAM/n
#             # ---- CAM
#             CAM_S = self.CAM(feat_S)
#             CAM_T = self.CAM(feat_T)
#             CAM_S = CAM_S.view(CAM_S.size(0), -1)
#             CAM_T = CAM_T.view(CAM_T.size(0), -1)
#             loss_CAM = self.linear_CKA_loss(CAM_S, CAM_T)
#             return loss_CAM, loss_PAM




#python
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class CriterionSA(nn.Module):
#     """
#     扩展的自注意蒸馏损失模块，修复 view 导致的 non-contiguous 问题（使用 reshape）。
#     """
#     def __init__(self,
#                  cwd_temperature=1.0,
#                  sa_temperature=1.0,
#                  num_masks=4,
#                  hidden_dim=64,
#                  align_method='cka',
#                  diversity_weight=0.1,
#                  max_spatial_size=64):
#         super(CriterionSA, self).__init__()
#         self.temperature = cwd_temperature
#         self.temperaturesa = sa_temperature
#         self.num_masks = int(num_masks)
#         self.hidden_dim = int(hidden_dim)
#         self.align_method = align_method
#         self.diversity_weight = float(diversity_weight)
#
#         self.kld = nn.KLDivLoss(reduction='mean')
#         self.mse = nn.MSELoss(reduction='mean')
#         self.softmax = nn.Softmax(dim=-1)
#         self.gammacam = nn.Parameter(torch.zeros(1))
#         self.gammapam = nn.Parameter(torch.zeros(1))
#
#         self._initialized = False
#         self.max_spatial_size = int(max_spatial_size)
#
#     def centering(self, K):
#         n = K.shape[0]
#         unit = K.new_ones((n, n))
#         I = torch.eye(n, device=K.device, dtype=K.dtype)
#         H = I - unit / n
#         return H @ K @ H
#
#     def linear_HSIC(self, X, Y):
#         L_X = X @ X.t()
#         L_Y = Y @ Y.t()
#         return torch.sum(self.centering(L_X) * self.centering(L_Y))
#
#     def linear_CKA_loss(self, X, Y):
#         hsic = self.linear_HSIC(X, Y)
#         var1 = torch.sqrt(self.linear_HSIC(X, X) + 1e-12)
#         var2 = torch.sqrt(self.linear_HSIC(Y, Y) + 1e-12)
#         cka = hsic / (var1 * var2 + 1e-12)
#         return -torch.log(torch.clamp(cka.mean(), min=1e-8))
#
#     def _lazy_init(self, C_s, C_t):
#         if self._initialized:
#             return
#         proj_dim = max(8, min(self.hidden_dim, C_s, C_t))
#         self.qs_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
#         self.ks_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
#         self.vs_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
#         self.qt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)
#         self.kt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)
#         self.vt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)
#         fuse_out = min(C_s, C_t, max(proj_dim, 16))
#         self.fuse_conv = nn.Conv2d(proj_dim * 2, fuse_out, kernel_size=1, bias=False)
#         self.mask_conv = nn.Conv2d(self.fuse_conv.out_channels, self.num_masks, kernel_size=1, bias=True)
#         self.cam_q = nn.Conv2d(self.fuse_conv.out_channels, max(1, self.fuse_conv.out_channels // 2), kernel_size=1, bias=False)
#         self.cam_k = nn.Conv2d(self.fuse_conv.out_channels, max(1, self.fuse_conv.out_channels // 2), kernel_size=1, bias=False)
#         self.pam_value = nn.Conv2d(self.fuse_conv.out_channels, self.fuse_conv.out_channels, kernel_size=1, bias=False)
#         self.align_s_conv = nn.Conv2d(C_s, self.fuse_conv.out_channels, kernel_size=1, bias=False)
#         self.align_t_conv = nn.Conv2d(C_t, self.fuse_conv.out_channels, kernel_size=1, bias=False)
#
#         self._initialized = True
#         try:
#             device = self.gammacam.device
#         except Exception:
#             device = torch.device('cpu')
#         self.to(device)
#
#     def _maybe_downsample(self, X):
#         B, C, H, W = X.shape
#         N = H * W
#         max_N = int(self.max_spatial_size) * int(self.max_spatial_size)
#         if N > max_N:
#             ds = self.max_spatial_size
#             X_ds = F.adaptive_avg_pool2d(X, (ds, ds))
#             return X_ds, (H, W), True
#         return X, (H, W), False
#
#     def CAM(self, X):
#         X_ds, (H, W), was_down = self._maybe_downsample(X)
#         B, C, Hd, Wd = X_ds.shape
#
#         q = self.cam_q(X_ds).view(B, -1, Hd * Wd)
#         k = self.cam_k(X_ds).view(B, -1, Hd * Wd)
#         energy = torch.bmm(q.permute(0, 2, 1), k)
#         energy_new = energy.max(-1, keepdim=True)[0].expand_as(energy) - energy
#         att = torch.softmax(energy_new / max(1e-6, float(self.temperaturesa)), dim=-1)
#         v = X_ds.view(B, C, -1).permute(0, 2, 1)
#         out = torch.bmm(att, v).permute(0, 2, 1).view(B, C, Hd, Wd)
#
#         if was_down:
#             out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
#         del q, k, energy, energy_new, att, v
#         return self.gammacam * out + X
#
#     def PAM(self, X):
#         X_ds, (H, W), was_down = self._maybe_downsample(X)
#         B, C, Hd, Wd = X_ds.shape
#
#         q = self.cam_q(X_ds).view(B, -1, Hd * Wd).permute(0, 2, 1)
#         k = self.cam_k(X_ds).view(B, -1, Hd * Wd)
#         energy = torch.bmm(q, k)
#         att = torch.softmax(energy / max(1e-6, float(self.temperaturesa)), dim=-1)
#         v = self.pam_value(X_ds).view(B, -1, Hd * Wd)
#         out = torch.bmm(v, att.permute(0, 2, 1)).view(B, C, Hd, Wd)
#
#         if was_down:
#             out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
#         del q, k, energy, att, v
#         return self.gammapam * out + X
#
#     def stca_ff(self, feat_S, feat_T):
#         B, Cs, H, W = feat_S.shape
#         _, Ct, _, _ = feat_T.shape
#         if not self._initialized:
#             self._lazy_init(Cs, Ct)
#
#         N = H * W
#         max_N = int(self.max_spatial_size) * int(self.max_spatial_size)
#         need_upsample = False
#         if N > max_N:
#             ds = int(self.max_spatial_size)
#             feat_S_ds = F.adaptive_avg_pool2d(feat_S, (ds, ds))
#             feat_T_ds = F.adaptive_avg_pool2d(feat_T, (ds, ds))
#             H_ds, W_ds = ds, ds
#             need_upsample = True
#         else:
#             feat_S_ds, feat_T_ds = feat_S, feat_T
#             H_ds, W_ds = H, W
#
#         B, _, _, _ = feat_S_ds.shape
#         qs = self.qs_conv(feat_S_ds).view(B, -1, H_ds * W_ds).permute(0, 2, 1)
#         kt = self.kt_conv(feat_T_ds).view(B, -1, H_ds * W_ds)
#         vt = self.vt_conv(feat_T_ds).view(B, -1, H_ds * W_ds).permute(0, 2, 1)
#
#         scale = max(1e-6, float(self.temperaturesa))
#         att_st = torch.softmax(torch.bmm(qs, kt) / scale, dim=-1)
#         out_st = torch.bmm(att_st, vt).permute(0, 2, 1).view(B, -1, H_ds, W_ds)
#
#         qt = self.qt_conv(feat_T_ds).view(B, -1, H_ds * W_ds).permute(0, 2, 1)
#         ks = self.ks_conv(feat_S_ds).view(B, -1, H_ds * W_ds)
#         vs = self.vs_conv(feat_S_ds).view(B, -1, H_ds * W_ds).permute(0, 2, 1)
#
#         att_ts = torch.softmax(torch.bmm(qt, ks) / scale, dim=-1)
#         out_ts = torch.bmm(att_ts, vs).permute(0, 2, 1).view(B, -1, H_ds, W_ds)
#
#         fused = torch.cat([out_st, out_ts], dim=1)
#         fused = self.fuse_conv(fused)
#
#         if need_upsample and (H_ds != H or W_ds != W):
#             fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
#
#         return fused
#
#     def generate_masks(self, fused):
#         logits = self.mask_conv(fused)
#         B, M, H, W = logits.shape
#         masks = F.softmax(logits.view(B, M, -1) / max(1e-6, float(self.temperaturesa)), dim=1).view(B, M, H, W)
#         return masks
#
#     def diversity_loss(self, masks):
#         B, M, H, W = masks.shape
#         m_flat = masks.view(B, M, -1)
#         m_mean = m_flat.mean(dim=0)
#         Mmat = m_mean
#         Mnorm = F.normalize(Mmat, p=2, dim=1)
#         G = Mnorm @ Mnorm.t()
#         off_diag = G - torch.diag(torch.diag(G))
#         loss = off_diag.sum() / (M * (M - 1) + 1e-12)
#         return loss
#
#     def forward(self, feat_S, feat_T, feature_transform="CAM_CKA", num_masks=None):
#         if num_masks is not None:
#             self.num_masks = int(num_masks)
#
#         B, Cs, H, W = feat_S.shape
#         _, Ct, _, _ = feat_T.shape
#
#         # 兼容旧模式（简化实现）
#         if feature_transform in ["CAM_CKA", "CAM_MSE", "PAM_CKA", "PAM_MSE",
#                                  "gridPAM_CKA", "gridPAM_MSE",
#                                  "separately_CAMgridPAM_CKA", "separately_CAMgridPAM_MSE"]:
#             if feature_transform == "CAM_CKA":
#                 if not self._initialized:
#                     self._lazy_init(Cs, Ct)
#                 S_mapped = self.align_s_conv(feat_S) if self.align_s_conv is not None else feat_S
#                 T_mapped = self.align_t_conv(feat_T) if self.align_t_conv is not None else feat_T
#                 CAM_S = self.CAM(S_mapped).reshape(B, -1)
#                 CAM_T = self.CAM(T_mapped).reshape(B, -1)
#                 return self.linear_CKA_loss(CAM_S, CAM_T)
#             if feature_transform == "CAM_MSE":
#                 if not self._initialized:
#                     self._lazy_init(Cs, Ct)
#                 CAM_S = self.CAM(self.align_s_conv(feat_S)).reshape(B, -1)
#                 CAM_T = self.CAM(self.align_t_conv(feat_T)).reshape(B, -1)
#                 return self.mse(CAM_S, CAM_T)
#             if feature_transform == "PAM_CKA":
#                 if not self._initialized:
#                     self._lazy_init(Cs, Ct)
#                 PAM_S = self.PAM(self.align_s_conv(feat_S)).reshape(B, -1)
#                 PAM_T = self.PAM(self.align_t_conv(feat_T)).reshape(B, -1)
#                 return self.linear_CKA_loss(PAM_S, PAM_T)
#             if feature_transform == "PAM_MSE":
#                 if not self._initialized:
#                     self._lazy_init(Cs, Ct)
#                 return self.mse(self.PAM(self.align_s_conv(feat_S)), self.PAM(self.align_t_conv(feat_T)))
#             if feature_transform == "gridPAM_CKA" or feature_transform == "gridPAM_MSE":
#                 if not self._initialized:
#                     self._lazy_init(Cs, Ct)
#                 S_mapped = self.align_s_conv(feat_S)
#                 firstpart_S = torch.chunk(S_mapped, 5, dim=2)
#                 partsS = []
#                 for i in firstpart_S:
#                     secondpart_S = torch.chunk(i, 5, dim=3)
#                     for j in secondpart_S:
#                         partsS.append(self.PAM(j))
#                 firstpart_T = torch.chunk(self.align_t_conv(feat_T), 5, dim=2)
#                 partsT = []
#                 for i in firstpart_T:
#                     secondpart_T = torch.chunk(i, 5, dim=3)
#                     for j in secondpart_T:
#                         partsT.append(self.PAM(j))
#                 loss = 0
#                 n = len(partsS)
#                 for i in range(n):
#                     if feature_transform == "gridPAM_CKA":
#                         S_pam = partsS[i].reshape(partsS[i].size(0), -1)
#                         T_pam = partsT[i].reshape(partsT[i].size(0), -1)
#                         loss += self.linear_CKA_loss(S_pam, T_pam)
#                     else:  # gridPAM_MSE
#                         loss += self.mse(partsS[i], partsT[i])
#                 return loss / max(1, n)
#             if feature_transform == "separately_CAMgridPAM_CKA" or feature_transform == "separately_CAMgridPAM_MSE":
#                 if not self._initialized:
#                     self._lazy_init(Cs, Ct)
#
#                 # 处理 CAM 部分
#                 S_mapped = self.align_s_conv(feat_S)
#                 T_mapped = self.align_t_conv(feat_T)
#                 CAM_S = self.CAM(S_mapped)
#                 CAM_T = self.CAM(T_mapped)
#
#                 # 处理 gridPAM 部分
#                 firstpart_S = torch.chunk(S_mapped, 5, dim=2)
#                 partsS = []
#                 for i in firstpart_S:
#                     secondpart_S = torch.chunk(i, 5, dim=3)
#                     for j in secondpart_S:
#                         partsS.append(self.PAM(j))
#
#                 firstpart_T = torch.chunk(T_mapped, 5, dim=2)
#                 partsT = []
#                 for i in firstpart_T:
#                     secondpart_T = torch.chunk(i, 5, dim=3)
#                     for j in secondpart_T:
#                         partsT.append(self.PAM(j))
#
#                 loss_PAM = 0
#                 n = len(partsS)
#                 for i in range(n):
#                     if feature_transform == "separately_CAMgridPAM_CKA":
#                         S_pam = partsS[i].reshape(partsS[i].size(0), -1)
#                         T_pam = partsT[i].reshape(partsT[i].size(0), -1)
#                         loss_PAM += self.linear_CKA_loss(S_pam, T_pam)
#                     else:  # separately_CAMgridPAM_MSE
#                         loss_PAM += self.mse(partsS[i], partsT[i])
#                 loss_PAM = loss_PAM / max(1, n)
#
#                 # 根据不同的方法计算 CAM 损失
#                 if feature_transform == "separately_CAMgridPAM_CKA":
#                     CAM_S_flat = CAM_S.reshape(CAM_S.size(0), -1)
#                     CAM_T_flat = CAM_T.reshape(CAM_T.size(0), -1)
#                     loss_CAM = self.linear_CKA_loss(CAM_S_flat, CAM_T_flat)
#                 else:  # separately_CAMgridPAM_MSE
#                     loss_CAM = self.mse(CAM_S, CAM_T)
#
#                 return loss_CAM, loss_PAM
#
#         # 新模式 STCA -> ASCM -> 多掩码对齐
#         if feature_transform in ["STCA_ASCM_M_CKA", "STCA_ASCM_M_MSE"]:
#             if not self._initialized:
#                 self._lazy_init(Cs, Ct)
#             fused = self.stca_ff(feat_S, feat_T)
#             masks = self.generate_masks(fused)
#             B, M, Hm, Wm = masks.shape
#
#             align_losses = []
#             for m in range(M):
#                 mask_m = masks[:, m:m+1, :, :]
#                 mapped_S = self.align_s_conv(feat_S)
#                 mapped_T = self.align_t_conv(feat_T)
#                 masked_S = mapped_S * (1.0 + mask_m)
#                 masked_T = mapped_T * mask_m
#                 cam_s = self.CAM(masked_S).reshape(B, -1)
#                 cam_t = self.CAM(masked_T).reshape(B, -1)
#                 pam_s = self.PAM(masked_S).reshape(B, -1)
#                 pam_t = self.PAM(masked_T).reshape(B, -1)
#                 if self.align_method == 'cka' or feature_transform.endswith("_CKA"):
#                     loss_cam = self.linear_CKA_loss(cam_s, cam_t)
#                     loss_pam = self.linear_CKA_loss(pam_s, pam_t)
#                 else:
#                     loss_cam = self.mse(cam_s, cam_t)
#                     loss_pam = self.mse(pam_s, pam_t)
#                 align_losses.append((loss_cam + loss_pam) * 0.5)
#
#             align_loss = torch.stack(align_losses).mean()
#             div_loss = self.diversity_loss(masks)
#             total_loss = align_loss + self.diversity_weight * div_loss
#             return total_loss
#
#         # fallback
#         if not self._initialized:
#             self._lazy_init(Cs, Ct)
#         CAM_S = self.CAM(self.align_s_conv(feat_S)).reshape(B, -1)
#         CAM_T = self.CAM(self.align_t_conv(feat_T)).reshape(B, -1)
#         return self.linear_CKA_loss(CAM_S, CAM_T)



# # python
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import matplotlib.pyplot as plt
# import os
#
# class CriterionSA(nn.Module):
#     """
#     自注意蒸馏损失（带 STCA -> ASCM -> 频域自适应多掩码对齐，SAMD）
#     - 在 STCA_ASCM_M_CKA 模式下：引入“按掩码自适应的频带选择 + 黎曼球频域 CKA 对齐”。
#     - 为避免显存不足：rFFT2 半谱 + 通道降维 + 特征自适应下采样（max_spatial_size）+ 按掩码循环即时释放。
#     """
#     def __init__(self,
#                  cwd_temperature=1.0,
#                  sa_temperature=1.0,
#                  num_masks=4,
#                  hidden_dim=64,
#                  align_method='cka',
#                  diversity_weight=0.1,
#                  max_spatial_size=64,
#                  lambda_freq=0.5):
#         super(CriterionSA, self).__init__()
#         self.temperature = cwd_temperature
#         self.temperaturesa = sa_temperature
#         self.num_masks = int(num_masks)
#         self.hidden_dim = int(hidden_dim)
#         self.align_method = align_method
#         self.diversity_weight = float(diversity_weight)
#         self.max_spatial_size = int(max_spatial_size)
#         # 空间/频域分支权重，建议 0.5~0.7 之间微调
#         self.lambda_freq = float(lambda_freq)
#
#         # 常用损失与参数
#         self.kld = nn.KLDivLoss(reduction='mean')
#         self.mse = nn.MSELoss(reduction='mean')
#         self.softmax = nn.Softmax(dim=-1)
#         self.gammacam = nn.Parameter(torch.zeros(1))
#         self.gammapam = nn.Parameter(torch.zeros(1))
#
#         # 延迟初始化（按输入通道数构建）
#         self._initialized = False
#
#         # 运行期缓存，避免重复构造网格
#         self._radial_cache = {}
#
#
#
#     # -------------------- 基础函数（CKA/居中） --------------------
#     def centering(self, K: torch.Tensor):
#         n = K.shape[0]
#         unit = K.new_ones((n, n))
#         I = torch.eye(n, device=K.device, dtype=K.dtype)
#         H = I - unit / n
#         return H @ K @ H
#
#     def linear_HSIC(self, X, Y):
#         L_X = X @ X.t()
#         L_Y = Y @ Y.t()
#         return torch.sum(self.centering(L_X) * self.centering(L_Y))
#
#     def linear_CKA_loss(self, X, Y):
#         # 输入形状：[B, D]
#         hsic = self.linear_HSIC(X, Y)
#         var1 = torch.sqrt(self.linear_HSIC(X, X) + 1e-12)
#         var2 = torch.sqrt(self.linear_HSIC(Y, Y) + 1e-12)
#         cka = hsic / (var1 * var2 + 1e-12)
#         return -torch.log(torch.clamp(cka.mean(), min=1e-8))
#
#     # -------------------- 延迟初始化：投影/对齐/掩码/注意力头 --------------------
#     def _lazy_init(self, C_s, C_t):
#         if self._initialized:
#             return
#         proj_dim = max(8, min(self.hidden_dim, C_s, C_t))
#
#         # STCA 所用投影（查询/键/值）
#         self.qs_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
#         self.ks_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
#         self.vs_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
#         self.qt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)
#         self.kt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)
#         self.vt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)
#
#         fuse_out = min(C_s, C_t, max(proj_dim, 16))
#
#         #self.fuse_conv = nn.Conv2d(proj_dim * 2, fuse_out, kernel_size=1, bias=False)
#
#         # 调整融合维度以匹配单向或双向输入
#         self.fuse_conv = nn.Conv2d(proj_dim * 2, fuse_out, kernel_size=1, bias=False)
#         self.fuse_conv_single = nn.Conv2d(proj_dim, fuse_out, kernel_size=1, bias=False)
#
#         self.mask_conv = nn.Conv2d(self.fuse_conv.out_channels, self.num_masks, kernel_size=1, bias=True)
#
#         # CAM/PAM 的轻量化通道投影
#         cam_dim = max(1, self.fuse_conv.out_channels // 2)
#         self.cam_q = nn.Conv2d(self.fuse_conv.out_channels, cam_dim, kernel_size=1, bias=False)
#         self.cam_k = nn.Conv2d(self.fuse_conv.out_channels, cam_dim, kernel_size=1, bias=False)
#         self.pam_value = nn.Conv2d(self.fuse_conv.out_channels, self.fuse_conv.out_channels, kernel_size=1, bias=False)
#
#         # S/T 对齐到共同维度
#         self.align_s_conv = nn.Conv2d(C_s, self.fuse_conv.out_channels, kernel_size=1, bias=False)
#         self.align_t_conv = nn.Conv2d(C_t, self.fuse_conv.out_channels, kernel_size=1, bias=False)
#
#         # 频域分支：通道降维后再做 rFFT（省显存）
#         spec_dim = max(8, min(32, self.fuse_conv.out_channels))
#         self.spectral_proj = nn.Conv2d(self.fuse_conv.out_channels, spec_dim, kernel_size=1, bias=False)
#
#         # 掩码自适应频带选择头：
#         # 输出 2 个标量：beta（控制带宽/平滑，映射到 (0.15,1.0)），gamma（混合低/高频，sigmoid 到 (0,1)）
#         self.freq_head = nn.Sequential(
#             #nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(self.fuse_conv.out_channels*2, 16, 1, bias=True),
#             nn.BatchNorm2d(16),  # 加个BN更稳
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 2, 1, bias=True)
#         )
#
#         self._initialized = True
#         # 将模块移动到当前参数所在设备
#         try:
#             device = self.gammacam.device
#         except Exception:
#             device = torch.device('cpu')
#         self.to(device)
#
#     # -------------------- 分辨率与下采样 --------------------
#     def _maybe_downsample(self, X):
#         B, C, H, W = X.shape
#         N = H * W
#         max_N = int(self.max_spatial_size) * int(self.max_spatial_size)
#         if N > max_N:
#             ds = int(self.max_spatial_size)
#             X_ds = F.adaptive_avg_pool2d(X, (ds, ds))
#             return X_ds, (H, W), True
#         return X, (H, W), False
#
#     # -------------------- CAM / PAM（保留以兼容旧路径） --------------------
#     def CAM(self, X):
#         X_ds, (H, W), was_down = self._maybe_downsample(X)
#         B, C, Hd, Wd = X_ds.shape
#         q = self.cam_q(X_ds).reshape(B, -1, Hd * Wd)
#         k = self.cam_k(X_ds).reshape(B, -1, Hd * Wd)
#         energy = torch.bmm(q.permute(0, 2, 1), k)
#         energy_new = energy.max(-1, keepdim=True)[0].expand_as(energy) - energy
#         att = torch.softmax(energy_new / max(1e-6, float(self.temperaturesa)), dim=-1)
#         v = X_ds.reshape(B, C, -1).permute(0, 2, 1)
#         out = torch.bmm(att, v).permute(0, 2, 1).reshape(B, C, Hd, Wd)
#         if was_down:
#             out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
#         del q, k, energy, energy_new, att, v
#         return self.gammacam * out + X
#
#     def PAM(self, X):
#         X_ds, (H, W), was_down = self._maybe_downsample(X)
#         B, C, Hd, Wd = X_ds.shape
#         q = self.cam_q(X_ds).reshape(B, -1, Hd * Wd).permute(0, 2, 1)
#         k = self.cam_k(X_ds).reshape(B, -1, Hd * Wd)
#         energy = torch.bmm(q, k)
#         att = torch.softmax(energy / max(1e-6, float(self.temperaturesa)), dim=-1)
#         v = self.pam_value(X_ds).reshape(B, -1, Hd * Wd)
#         out = torch.bmm(v, att.permute(0, 2, 1)).reshape(B, C, Hd, Wd)
#         if was_down:
#             out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
#         del q, k, energy, att, v
#         return self.gammapam * out + X
#
#     # -------------------- STCA 融合 --------------------
#     def stca_ff(self, feat_S, feat_T, activate_s_to_t=True):
#         B, Cs, H, W = feat_S.shape
#         _, Ct, _, _ = feat_T.shape
#         if not self._initialized:
#             self._lazy_init(Cs, Ct)
#
#         # 下采样以避免显存压力
#         N = H * W
#         max_N = int(self.max_spatial_size) * int(self.max_spatial_size)
#         need_upsample = False
#         if N > max_N:
#             ds = int(self.max_spatial_size)
#             feat_S_ds = F.adaptive_avg_pool2d(feat_S, (ds, ds))
#             feat_T_ds = F.adaptive_avg_pool2d(feat_T, (ds, ds))
#             H_ds, W_ds = ds, ds
#             need_upsample = True
#         else:
#             feat_S_ds, feat_T_ds = feat_S, feat_T
#             H_ds, W_ds = H, W
#
#         B = feat_S_ds.shape[0]
#         scale = max(1e-6, float(self.temperaturesa))
#
#         # T->S (Teacher to Student) path is always active
#         qt = self.qt_conv(feat_T_ds).reshape(B, -1, H_ds * W_ds).permute(0, 2, 1)
#         ks = self.ks_conv(feat_S_ds).reshape(B, -1, H_ds * W_ds)
#         vs = self.vs_conv(feat_S_ds).reshape(B, -1, H_ds * W_ds).permute(0, 2, 1)
#
#         att_ts = torch.softmax(torch.bmm(qt, ks) / scale, dim=-1)
#         out_ts = torch.bmm(att_ts, vs).permute(0, 2, 1).reshape(B, -1, H_ds, W_ds)
#
#         if activate_s_to_t:
#             # S->T (Student to Teacher) path, conditionally activated
#             qs = self.qs_conv(feat_S_ds).reshape(B, -1, H_ds * W_ds).permute(0, 2, 1)
#             kt = self.kt_conv(feat_T_ds).reshape(B, -1, H_ds * W_ds)
#             vt = self.vt_conv(feat_T_ds).reshape(B, -1, H_ds * W_ds).permute(0, 2, 1)
#
#             att_st = torch.softmax(torch.bmm(qs, kt) / scale, dim=-1)
#             out_st = torch.bmm(att_st, vt).permute(0, 2, 1).reshape(B, -1, H_ds, W_ds)
#
#             fused = torch.cat([out_st, out_ts], dim=1)
#             fused = self.fuse_conv(fused)
#         else:
#             fused = self.fuse_conv_single(out_ts)
#
#
#         if need_upsample and (H_ds != H or W_ds != W):
#             fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
#         return fused
#
#     def generate_masks(self, fused):
#         logits = self.mask_conv(fused)
#         B, M, H, W = logits.shape
#         masks = F.softmax(logits.reshape(B, M, -1) / max(1e-6, float(self.temperaturesa)), dim=1).reshape(B, M, H, W)
#         return masks
#
#     def diversity_loss(self, masks):
#         # 鼓励不同掩码关注不重叠区域（简单的余弦相似度正则）
#         B, M, H, W = masks.shape
#         m_flat = masks.reshape(B, M, -1)  # [B, M, HW]
#         m_mean = m_flat.mean(dim=0)       # [M, HW]
#         m_norm = F.normalize(m_mean, p=2, dim=1)
#         G = m_norm @ m_norm.t()           # [M, M]
#         off = G - torch.diag(torch.diag(G))
#         return off.sum() / (M * (M - 1) + 1e-12)
#
#     # -------------------- SAMD：频域模块 --------------------
#     @staticmethod
#     def _riemann_map(real: torch.Tensor, imag: torch.Tensor):
#         """
#         将复平面映射到单位黎曼球坐标，稳定比较相位/幅度。
#         输入/输出形状均与 real/imag 相同（返回 3 个分量列表）。
#         """
#         deno = (real * real + imag * imag + 1).clamp_min(1e-9)
#         xR = (2 * real) / deno
#         yR = (2 * imag) / deno
#         zR = (real * real + imag * imag - 1) / deno
#         return [xR, yR, zR]
#
#     def _get_radial_weight(self, Hf, Wf, device, dtype):
#         """
#         构造以(0,0)为低频原点的半谱径向半径（r ∈ [0,1]），缓存避免重复构造。
#         注意：rFFT2 输出为半谱，横轴长度为 Wf，仍可用归一化网格生成半径。
#         """
#         key = (Hf, Wf, str(device), str(dtype))
#         if key in self._radial_cache:
#             return self._radial_cache[key]
#         yy = torch.linspace(0, 1, steps=Hf, device=device, dtype=dtype).view(Hf, 1).expand(Hf, Wf)
#         xx = torch.linspace(0, 1, steps=Wf, device=device, dtype=dtype).view(1, Wf).expand(Hf, Wf)
#         r = torch.sqrt(xx * xx + yy * yy)  # 原点(0,0)处为最低频
#         r = r.clamp_max(1.0)
#         weight = r.unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
#         self._radial_cache[key] = weight
#         return weight
#
#     # def _freq_weight_from_context(self, ctx_feat, mask_m):
#     #     """
#     #     从融合特征 + 掩码得到频带参数：
#     #     - beta: 控制带宽（映射到 σ ∈ (0.15, 1.0) 的高斯低通）
#     #     - gamma: 低频/高频混合因子（sigmoid 到 (0,1)）
#     #     返回：sigma[b,1,1,1], gamma[b,1,1,1]
#     #     """
#     #     # 掩码化上下文，提升可分辨的局部结构感知
#     #     ctx = ctx_feat * mask_m
#     #     params = self.freq_head(ctx)  # [B,2,1,1]
#     #     beta = torch.sigmoid(params[:, 0:1, ...])  # (0,1)
#     #     gamma = torch.sigmoid(params[:, 1:2, ...]) # (0,1)
#     #     sigma = 0.15 + 0.85 * beta                 # 避免过窄带
#     #     return sigma, gamma
#
#     def _freq_weight_from_context(self, ctx_feat, mask_m):
#         # ctx_feat: [B, C, H, W]
#         # mask_m:   [B, 1, H, W]
#
#         # 1. 提取感兴趣区域 (ROI)
#         roi = ctx_feat * mask_m
#
#         # 计算每个 Mask 的有效面积 (防止除以 0)
#         # sum(dim=(2,3)) 表示在 H, W 维度求和
#         mask_area = mask_m.sum(dim=(2, 3), keepdim=True).clamp_min(1e-5)  # [B, 1, 1, 1]
#
#         # 2. 计算【掩码区域内】的均值 (Masked Mean) -> 代表语义
#         # Sum(Features) / Sum(Mask)
#         roi_sum = roi.sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
#         feat_mean = roi_sum / mask_area
#
#         # 3. 计算【掩码区域内】的标准差 (Masked Std) -> 代表纹理复杂度
#         # Std 是区分平滑(天空)和纹理(树)的最强特征
#         # var = E[x^2] - (E[x])^2
#         roi_sq_sum = (roi ** 2).sum(dim=(2, 3), keepdim=True)
#         feat_sq_mean = roi_sq_sum / mask_area
#         # 这里的 clamp 也就是为了数值稳定，防止负数开根号
#         feat_var = (feat_sq_mean - feat_mean ** 2).clamp_min(1e-5)
#         feat_std = torch.sqrt(feat_var)
#
#         # 4. 拼接：均值 + 标准差
#         # [B, 2C, 1, 1]
#         feat_cat = torch.cat([feat_mean, feat_std], dim=1)
#
#         # 5. 预测参数
#         params = self.freq_head(feat_cat)
#
#         beta = torch.sigmoid(params[:, 0:1, ...])
#         gamma = torch.sigmoid(params[:, 1:2, ...])
#         sigma = 0.15 + 0.85 * beta
#
#         # ================= [再次修改这里] =================
#         # 为了可视化区分度，我们采取更激进的策略：
#         # 记录每个样本在所有通道中的【最大标准差】(Max Std)
#         # B: Batch Size (Sentinel时为1)
#
#         # 1. 展平通道维度 [B, C]
#         std_flat = feat_std.view(feat_std.size(0), -1)
#         mean_flat = feat_mean.view(feat_mean.size(0), -1)
#
#         # 2. 取最大值 (Max)，而不是均值
#         # 这样能捕捉到“最显著的纹理特征”
#         representative_std, _ = std_flat.max(dim=1)  # [B]
#
#         # 对于均值，取绝对值的最大值 (因为有正负)
#         representative_mean, _ = mean_flat.abs().max(dim=1)  # [B]
#
#         # 调试用：如果您想确认 feat_std 是否真的拉开了差距，可以取消下面的注释
#         # if not self.training:
#         #     print(f"Mask Mean: {feat_mean.mean().item():.3f}, Mask Std: {feat_std.mean().item():.3f}")
#
#         # [修改] 返回值增加 feat_mean 和 feat_std (取均值变成标量)
#         # 注意：这里返回的是 Tensor，带梯度的
#         return sigma, gamma, representative_mean, representative_std
#
#     # def _fft_riemann_and_weighted_cka(self, S_feat, T_feat, ctx_feat, mask_m, return_debug=False):
#     #     """
#     #             针对单个掩码，执行：
#     #             1) 通道降维 -> rFFT2 -> 黎曼映射
#     #             2) 用 sigma/gamma 生成频带权重（低/高频混合）
#     #             3) 加权（按频点）后的 CKA（对三维黎曼坐标平均）
#     #             返回：标量 loss
#     #             """
#     #     """
#     #             return_debug=True: 返回 (loss, sigma, gamma, Filter_W, freq_loss_val)
#     #             return_debug=False: 返回 (loss, sigma, gamma)
#     #             """
#     #
#     #
#     #     # 教师频域分支不回传梯度，降低显存占用
#     #     T_feat = T_feat.detach()
#     #
#     #     # 将 S/T 对齐到共同维度并通道降维以节省显存
#     #     S_m = self.spectral_proj(S_feat)
#     #     T_m = self.spectral_proj(T_feat)
#     #
#     #     # 掩码增强：学生乘 (1+mask)，教师乘 mask，鼓励学生在掩码区域“追老师”
#     #     S_m = S_m * (1.0 + mask_m)
#     #     T_m = T_m * mask_m
#     #
#     #     # # 下采样（频域计算分辨率不宜过大）
#     #     # S_ds, (Hs, Ws), _ = self._maybe_downsample(S_m)
#     #     # T_ds, _, _ = self._maybe_downsample(T_m)
#     #
#     #     # rFFT2：半谱，显存更低；需要同一分辨率
#     #     F_S = torch.fft.rfft2(S_m, dim=(2, 3), norm='ortho')  # [B,C,Hf,Wf']
#     #     F_T = torch.fft.rfft2(T_m, dim=(2, 3), norm='ortho')
#     #     real_S, imag_S = F_S.real, F_S.imag
#     #     real_T, imag_T = F_T.real, F_T.imag
#     #
#     #     # 黎曼球映射，得到三分量
#     #     R_S = self._riemann_map(real_S, imag_S)  # list of 3 tensors
#     #     R_T = self._riemann_map(real_T, imag_T)
#     #
#     #     # 基于上下文 + 掩码，得到频带参数并生成权重图
#     #     sigma, gamma = self._freq_weight_from_context(ctx_feat, mask_m)  # [B,1,1,1]
#     #
#     #     B, C, Hf, Wf = real_S.shape
#     #     radial = self._get_radial_weight(Hf, Wf, device=real_S.device, dtype=real_S.dtype)  # [1,1,Hf,Wf]
#     #     # 低通（高斯）与高通（互补）
#     #     w_low = torch.exp(-(radial * radial) / (2.0 * (sigma ** 2)))   # [B,1,Hf,Wf]（广播）
#     #     w_high = 1.0 - w_low
#     #     W = gamma * w_high + (1.0 - gamma) * w_low   # [B,1,Hf,Wf]
#     #     sqrtW = torch.sqrt(W.clamp_min(1e-8))
#     #
#     #     # 对每个黎曼分量进行加权，随后展平到 [B, D] 用 CKA
#     #     losses = []
#     #     for s_comp, t_comp in zip(R_S, R_T):
#     #         # [B,C,Hf,Wf] * [B,1,Hf,Wf] -> [B,C,Hf,Wf]
#     #         s_w = (s_comp * sqrtW).reshape(B, -1)
#     #         t_w = (t_comp * sqrtW).reshape(B, -1)
#     #         losses.append(self.linear_CKA_loss(s_w, t_w))
#     #     loss = sum(losses) / 3.0
#     #
#     #     # 及时释放中间变量，降低峰值显存
#     #     del F_S, F_T, real_S, imag_S, real_T, imag_T, R_S, R_T, w_low, w_high, W, sqrtW
#     #
#     #     if torch.cuda.is_available():
#     #         torch.cuda.empty_cache()
#     #
#     #     if return_debug:
#     #         # 返回: Loss, Sigma, Gamma, FilterMap(W)
#     #         return loss, sigma, gamma, W
#     #     else:
#     #         return loss, sigma, gamma
#     #
#     #     # return loss, sigma, gamma
#
#     def _fft_riemann_and_weighted_cka(self, S_feat, T_feat, ctx_feat, mask_m, return_debug=False):
#         """
#         针对单个掩码，执行：
#         1) 通道降维 -> rFFT2 -> 黎曼映射
#         2) 用 sigma/gamma 生成频带权重（低/高频混合）
#         3) 加权（按频点）后的 CKA（对三维黎曼坐标平均）
#         返回：标量 loss
#         """
#
#         # 教师频域分支不回传梯度，降低显存占用
#         T_feat = T_feat.detach()
#
#         # 将 S/T 对齐到共同维度并通道降维以节省显存
#         S_m = self.spectral_proj(S_feat)
#         T_m = self.spectral_proj(T_feat)
#
#         # 掩码增强：学生乘 (1+mask)，教师乘 mask，鼓励学生在掩码区域“追老师”
#         S_m = S_m * (1.0 + mask_m)
#         T_m = T_m * mask_m
#
#         # rFFT2：半谱，显存更低；需要同一分辨率
#         F_S = torch.fft.rfft2(S_m, dim=(2, 3), norm='ortho')  # [B,C,Hf,Wf']
#         F_T = torch.fft.rfft2(T_m, dim=(2, 3), norm='ortho')
#         real_S, imag_S = F_S.real, F_S.imag
#         real_T, imag_T = F_T.real, F_T.imag
#
#         # 黎曼球映射，得到三分量
#         R_S = self._riemann_map(real_S, imag_S)  # list of 3 tensors
#         R_T = self._riemann_map(real_T, imag_T)
#
#         # 基于上下文 + 掩码，得到频带参数并生成权重图
#         #sigma, gamma = self._freq_weight_from_context(ctx_feat, mask_m)  # [B,1,1,1]
#         sigma, gamma, f_mean, f_std = self._freq_weight_from_context(ctx_feat, mask_m)
#
#         B, C, Hf, Wf = real_S.shape
#         radial = self._get_radial_weight(Hf, Wf, device=real_S.device, dtype=real_S.dtype)  # [1,1,Hf,Wf]
#
#         # 低通（高斯）与高通（互补）
#         w_low = torch.exp(-(radial * radial) / (2.0 * (sigma ** 2)))  # [B,1,Hf,Wf]（广播）
#         w_high = 1.0 - w_low
#         W = gamma * w_high + (1.0 - gamma) * w_low  # [B,1,Hf,Wf]
#         sqrtW = torch.sqrt(W.clamp_min(1e-8))
#
#         # ==================== [核心修改开始] ====================
#         losses = []
#
#         # 判断当前是 正常训练(B>1) 还是 可视化监测(B=1)
#         if B == 1:
#             # [新增逻辑] 针对单张图片的特殊处理
#             # 这里的 loss 是为了画图展示 "距离在缩小"，MSE 是最佳替代品
#             for s_comp, t_comp in zip(R_S, R_T):
#                 # 直接计算加权后的特征距离
#                 s_w = s_comp * sqrtW
#                 t_w = t_comp * sqrtW
#                 losses.append(self.mse(s_w, t_w))
#             loss = sum(losses)
#         else:
#             # [原有逻辑] 正常的 CKA 计算（用于训练，需要 Batch 统计特性）
#             for s_comp, t_comp in zip(R_S, R_T):
#                 # [B,C,Hf,Wf] * [B,1,Hf,Wf] -> [B,C,Hf,Wf]
#                 s_w = (s_comp * sqrtW).reshape(B, -1)
#                 t_w = (t_comp * sqrtW).reshape(B, -1)
#                 losses.append(self.linear_CKA_loss(s_w, t_w))
#             loss = sum(losses) / 3.0
#         # ==================== [核心修改结束] ====================
#
#         # 及时释放中间变量，降低峰值显存
#         # 注意：这里千万不要 del W，因为下面 return_debug 要用到它
#         del F_S, F_T, real_S, imag_S, real_T, imag_T, R_S, R_T, w_low, w_high, sqrtW
#
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         if return_debug:
#             # 返回: Loss, Sigma, Gamma, FilterMap(W)
#             return loss, sigma, gamma, W, f_mean, f_std
#         else:
#             return loss, sigma, gamma
#
#
#     # -------------------- 前向 --------------------
#     def forward(self, feat_S, feat_T, feature_transform="CAM_CKA", num_masks=None, activate_s_to_t=True, visualization_mode=False):
#         if num_masks is not None:
#             self.num_masks = int(num_masks)
#
#         B, Cs, H, W = feat_S.shape
#         _, Ct, _, _ = feat_T.shape
#         # if not self._initialized:
#         #     self._lazy_init(Cs, Ct)
#
#         # 教师侧不回传梯度，避免保存教师计算图
#         feat_T = feat_T.detach()
#
#         # 兼容旧模式（保持不变）
#         if feature_transform in ["CAM_CKA", "CAM_MSE", "PAM_CKA", "PAM_MSE",
#                                  "gridPAM_CKA", "gridPAM_MSE",
#                                  "separately_CAMgridPAM_CKA", "separately_CAMgridPAM_MSE"]:
#             # ...... 保持原实现（略） ......
#             # 为简洁起见，此处应复用你现有代码的对应分支。
#             S_mapped = self.align_s_conv(feat_S)
#             T_mapped = self.align_t_conv(feat_T)
#             if feature_transform == "CAM_CKA":
#                 CAM_S = self.CAM(S_mapped).reshape(B, -1)
#                 CAM_T = self.CAM(T_mapped).reshape(B, -1)
#                 return self.linear_CKA_loss(CAM_S, CAM_T)
#             if feature_transform == "CAM_MSE":
#                 return self.mse(self.CAM(S_mapped), self.CAM(T_mapped))
#             if feature_transform == "PAM_CKA":
#                 PAM_S = self.PAM(S_mapped).reshape(B, -1)
#                 PAM_T = self.PAM(T_mapped).reshape(B, -1)
#                 return self.linear_CKA_loss(PAM_S, PAM_T)
#             if feature_transform == "PAM_MSE":
#                 return self.mse(self.PAM(S_mapped), self.PAM(T_mapped))
#             if feature_transform in ["gridPAM_CKA", "gridPAM_MSE"]:
#                 firstpart_S = torch.chunk(S_mapped, 5, dim=2)
#                 partsS = []
#                 for i in firstpart_S:
#                     secondpart_S = torch.chunk(i, 5, dim=3)
#                     for j in secondpart_S:
#                         partsS.append(self.PAM(j))
#                 firstpart_T = torch.chunk(T_mapped, 5, dim=2)
#                 partsT = []
#                 for i in firstpart_T:
#                     secondpart_T = torch.chunk(i, 5, dim=3)
#                     for j in secondpart_T:
#                         partsT.append(self.PAM(j))
#                 loss = 0
#                 n = len(partsS)
#                 for i in range(n):
#                     if feature_transform == "gridPAM_CKA":
#                         S_p = partsS[i].reshape(partsS[i].size(0), -1)
#                         T_p = partsT[i].reshape(partsT[i].size(0), -1)
#                         loss += self.linear_CKA_loss(S_p, T_p)
#                     else:
#                         loss += self.mse(partsS[i], partsT[i])
#                 return loss / max(1, n)
#             if feature_transform in ["separately_CAMgridPAM_CKA", "separately_CAMgridPAM_MSE"]:
#                 CAM_S = self.CAM(S_mapped)
#                 CAM_T = self.CAM(T_mapped)
#                 firstpart_S = torch.chunk(S_mapped, 5, dim=2)
#                 partsS = []
#                 for i in firstpart_S:
#                     secondpart_S = torch.chunk(i, 5, dim=3)
#                     for j in secondpart_S:
#                         partsS.append(self.PAM(j))
#                 firstpart_T = torch.chunk(T_mapped, 5, dim=2)
#                 partsT = []
#                 for i in firstpart_T:
#                     secondpart_T = torch.chunk(i, 5, dim=3)
#                     for j in secondpart_T:
#                         partsT.append(self.PAM(j))
#                 loss_P = 0
#                 n = len(partsS)
#                 for i in range(n):
#                     if feature_transform == "separately_CAMgridPAM_CKA":
#                         S_p = partsS[i].reshape(partsS[i].size(0), -1)
#                         T_p = partsT[i].reshape(partsT[i].size(0), -1)
#                         loss_P += self.linear_CKA_loss(S_p, T_p)
#                     else:
#                         loss_P += self.mse(partsS[i], partsT[i])
#                 loss_P = loss_P / max(1, n)
#                 if feature_transform == "separately_CAMgridPAM_CKA":
#                     return self.linear_CKA_loss(CAM_S.reshape(B, -1), CAM_T.reshape(B, -1)), loss_P
#                 else:
#                     return self.mse(CAM_S, CAM_T), loss_P
#
#         # -------------------- 新模式：STCA_ASCM_M_CKA -> SAMD 频域多掩码对齐 --------------------
#         if feature_transform in ["STCA_ASCM_M_CKA", "STCA_ASCM_M_MSE"]:
#             # 1) STCA 融合得到上下文
#             fused = self.stca_ff(feat_S, feat_T, activate_s_to_t=activate_s_to_t)                      # [B,F,H,W]
#             masks = self.generate_masks(fused)                        # [B,M,H,W]
#             S_al = self.align_s_conv(feat_S)                          # [B,F,H,W]
#             T_al = self.align_t_conv(feat_T)                          # [B,F,H,W]
#
#             # ================= [可视化模式] =================
#             # 如果开启，直接返回第一个 mask 的详细信息用于绘图
#             if visualization_mode:
#                 # mask_m = masks[:, 0:1, :, :]  # 取第1个mask做演示
#                 # freq_loss, s_val, g_val, W_map, f_mean, f_std = self._fft_riemann_and_weighted_cka(
#                 #     S_feat=S_al, T_feat=T_al, ctx_feat=fused, mask_m=mask_m, return_debug=False
#                 # )
#                 # # 返回:
#                 # return mask_m, W_map, s_val, g_val, freq_loss, f_mean, f_std
#                 return masks
#
#             # ================= [正常训练模式] =================
#             per_mask_losses = []
#             M = masks.shape[1]
#
#             total_sigma = 0
#             total_gamma = 0
#
#             for m in range(M):
#                 mask_m = masks[:, m:m+1, :, :]                        # [B,1,H,W]
#
#                 # 空间分支：CAM + PAM
#                 S_masked = S_al * (1.0 + mask_m)
#                 T_masked = T_al * mask_m
#
#                 cam_s = self.CAM(S_masked).reshape(B, -1)
#                 pam_s = self.PAM(S_masked).reshape(B, -1)
#
#
#                 with torch.no_grad():
#                     cam_t = self.CAM(T_masked).reshape(B, -1)
#                     pam_t = self.PAM(T_masked).reshape(B, -1)
#
#                 if self.align_method == 'cka' or feature_transform.endswith("_CKA"):
#                     loss_cam = self.linear_CKA_loss(cam_s, cam_t)
#                     loss_pam = self.linear_CKA_loss(pam_s, pam_t)
#                     spatial_loss = 0.5 * (loss_cam + loss_pam)
#                 else:
#                     spatial_loss = 0.5 * (self.mse(cam_s, cam_t) + self.mse(pam_s, pam_t))
#
#                 # 频域分支：SAMD
#                 # SAMD：在频域内做按掩码自适应的加权 CKA
#                 # freq_loss = self._fft_riemann_and_weighted_cka(
#                 #     S_feat=S_al, T_feat=T_al, ctx_feat=fused, mask_m=mask_m
#                 # )
#
#                 freq_loss, sigma_val, gamma_val = self._fft_riemann_and_weighted_cka(
#                     S_feat=S_al, T_feat=T_al, ctx_feat=fused, mask_m=mask_m
#                 )
#
#                 # 记录均值供 Tensorboard 使用
#                 total_sigma += sigma_val.mean()
#                 total_gamma += gamma_val.mean()
#
#                 # 融合（默认等权，可按需调整 lambda_freq）
#                 loss_m = (1.0 - self.lambda_freq) * spatial_loss + self.lambda_freq * freq_loss
#                 per_mask_losses.append(loss_m)
#
#
#             align_loss = torch.stack(per_mask_losses).mean()
#             div_loss = self.diversity_loss(masks)
#             total = align_loss + self.diversity_weight * div_loss
#
#             # 返回 Loss 和 (平均sigma, 平均gamma)
#             return total, (total_sigma / M, total_gamma / M)
#             # return total, freq_loss
#
#         # 兜底：默认 CAM_CKA
#         CAM_S = self.CAM(self.align_s_conv(feat_S)).reshape(B, -1)
#         CAM_T = self.CAM(self.align_t_conv(feat_T)).reshape(B, -1)
#         return self.linear_CKA_loss(CAM_S, CAM_T)



#用于可视化mask->std->两个频域参数关系---train_sa._vis_2.py
# python
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os

class CriterionSA(nn.Module):
    """
    自注意蒸馏损失（带 STCA -> ASCM -> 频域自适应多掩码对齐，SAMD）
    - 在 STCA_ASCM_M_CKA 模式下：引入“按掩码自适应的频带选择 + 黎曼球频域 CKA 对齐”。
    - 为避免显存不足：rFFT2 半谱 + 通道降维 + 特征自适应下采样（max_spatial_size）+ 按掩码循环即时释放。
    """
    def __init__(self,
                 cwd_temperature=1.0,
                 sa_temperature=1.0,
                 num_masks=4,
                 hidden_dim=64,
                 align_method='cka',
                 diversity_weight=0.1,
                 max_spatial_size=64,
                 lambda_freq=0.5):
        super(CriterionSA, self).__init__()
        self.temperature = cwd_temperature
        self.temperaturesa = sa_temperature
        self.num_masks = int(num_masks)
        self.hidden_dim = int(hidden_dim)
        self.align_method = align_method
        self.diversity_weight = float(diversity_weight)
        self.max_spatial_size = int(max_spatial_size)
        # 空间/频域分支权重，建议 0.5~0.7 之间微调
        self.lambda_freq = float(lambda_freq)

        # 常用损失与参数
        self.kld = nn.KLDivLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
        self.softmax = nn.Softmax(dim=-1)
        self.gammacam = nn.Parameter(torch.zeros(1))
        self.gammapam = nn.Parameter(torch.zeros(1))

        # 延迟初始化（按输入通道数构建）
        self._initialized = False

        # 运行期缓存，避免重复构造网格
        self._radial_cache = {}



    # -------------------- 基础函数（CKA/居中） --------------------
    def centering(self, K: torch.Tensor):
        n = K.shape[0]
        unit = K.new_ones((n, n))
        I = torch.eye(n, device=K.device, dtype=K.dtype)
        H = I - unit / n
        return H @ K @ H

    def linear_HSIC(self, X, Y):
        L_X = X @ X.t()
        L_Y = Y @ Y.t()
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA_loss(self, X, Y):
        # 输入形状：[B, D]
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X) + 1e-12)
        var2 = torch.sqrt(self.linear_HSIC(Y, Y) + 1e-12)
        cka = hsic / (var1 * var2 + 1e-12)
        return -torch.log(torch.clamp(cka.mean(), min=1e-8))

    # -------------------- 延迟初始化：投影/对齐/掩码/注意力头 --------------------
    def _lazy_init(self, C_s, C_t):
        if self._initialized:
            return
        proj_dim = max(8, min(self.hidden_dim, C_s, C_t))

        # STCA 所用投影（查询/键/值）
        self.qs_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
        self.ks_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
        self.vs_conv = nn.Conv2d(C_s, proj_dim, kernel_size=1, bias=False)
        self.qt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)
        self.kt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)
        self.vt_conv = nn.Conv2d(C_t, proj_dim, kernel_size=1, bias=False)

        fuse_out = min(C_s, C_t, max(proj_dim, 16))

        #self.fuse_conv = nn.Conv2d(proj_dim * 2, fuse_out, kernel_size=1, bias=False)

        # 调整融合维度以匹配单向或双向输入
        self.fuse_conv = nn.Conv2d(proj_dim * 2, fuse_out, kernel_size=1, bias=False)
        self.fuse_conv_single = nn.Conv2d(proj_dim, fuse_out, kernel_size=1, bias=False)

        self.mask_conv = nn.Conv2d(self.fuse_conv.out_channels, self.num_masks, kernel_size=1, bias=True)

        # CAM/PAM 的轻量化通道投影
        cam_dim = max(1, self.fuse_conv.out_channels // 2)
        self.cam_q = nn.Conv2d(self.fuse_conv.out_channels, cam_dim, kernel_size=1, bias=False)
        self.cam_k = nn.Conv2d(self.fuse_conv.out_channels, cam_dim, kernel_size=1, bias=False)
        self.pam_value = nn.Conv2d(self.fuse_conv.out_channels, self.fuse_conv.out_channels, kernel_size=1, bias=False)

        # S/T 对齐到共同维度
        self.align_s_conv = nn.Conv2d(C_s, self.fuse_conv.out_channels, kernel_size=1, bias=False)
        self.align_t_conv = nn.Conv2d(C_t, self.fuse_conv.out_channels, kernel_size=1, bias=False)

        # 频域分支：通道降维后再做 rFFT（省显存）
        spec_dim = max(8, min(32, self.fuse_conv.out_channels))
        self.spectral_proj = nn.Conv2d(self.fuse_conv.out_channels, spec_dim, kernel_size=1, bias=False)

        # 掩码自适应频带选择头：
        # 输出 2 个标量：beta（控制带宽/平滑，映射到 (0.15,1.0)），gamma（混合低/高频，sigmoid 到 (0,1)）
        self.freq_head = nn.Sequential(
            #nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.fuse_conv.out_channels*2, 16, 1, bias=True),
            nn.BatchNorm2d(16),  # 加个BN更稳
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1, bias=True)
        )

        self._initialized = True
        # 将模块移动到当前参数所在设备
        try:
            device = self.gammacam.device
        except Exception:
            device = torch.device('cpu')
        self.to(device)

    # -------------------- 分辨率与下采样 --------------------
    def _maybe_downsample(self, X):
        B, C, H, W = X.shape
        N = H * W
        max_N = int(self.max_spatial_size) * int(self.max_spatial_size)
        if N > max_N:
            ds = int(self.max_spatial_size)
            X_ds = F.adaptive_avg_pool2d(X, (ds, ds))
            return X_ds, (H, W), True
        return X, (H, W), False

    # -------------------- CAM / PAM（保留以兼容旧路径） --------------------
    def CAM(self, X):
        X_ds, (H, W), was_down = self._maybe_downsample(X)
        B, C, Hd, Wd = X_ds.shape
        q = self.cam_q(X_ds).reshape(B, -1, Hd * Wd)
        k = self.cam_k(X_ds).reshape(B, -1, Hd * Wd)
        energy = torch.bmm(q.permute(0, 2, 1), k)
        energy_new = energy.max(-1, keepdim=True)[0].expand_as(energy) - energy
        att = torch.softmax(energy_new / max(1e-6, float(self.temperaturesa)), dim=-1)
        v = X_ds.reshape(B, C, -1).permute(0, 2, 1)
        out = torch.bmm(att, v).permute(0, 2, 1).reshape(B, C, Hd, Wd)
        if was_down:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        del q, k, energy, energy_new, att, v
        return self.gammacam * out + X

    def PAM(self, X):
        X_ds, (H, W), was_down = self._maybe_downsample(X)
        B, C, Hd, Wd = X_ds.shape
        q = self.cam_q(X_ds).reshape(B, -1, Hd * Wd).permute(0, 2, 1)
        k = self.cam_k(X_ds).reshape(B, -1, Hd * Wd)
        energy = torch.bmm(q, k)
        att = torch.softmax(energy / max(1e-6, float(self.temperaturesa)), dim=-1)
        v = self.pam_value(X_ds).reshape(B, -1, Hd * Wd)
        out = torch.bmm(v, att.permute(0, 2, 1)).reshape(B, C, Hd, Wd)
        if was_down:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        del q, k, energy, att, v
        return self.gammapam * out + X

    # -------------------- STCA 融合 --------------------
    def stca_ff(self, feat_S, feat_T, activate_s_to_t=True):
        B, Cs, H, W = feat_S.shape
        _, Ct, _, _ = feat_T.shape
        if not self._initialized:
            self._lazy_init(Cs, Ct)

        # 下采样以避免显存压力
        N = H * W
        max_N = int(self.max_spatial_size) * int(self.max_spatial_size)
        need_upsample = False
        if N > max_N:
            ds = int(self.max_spatial_size)
            feat_S_ds = F.adaptive_avg_pool2d(feat_S, (ds, ds))
            feat_T_ds = F.adaptive_avg_pool2d(feat_T, (ds, ds))
            H_ds, W_ds = ds, ds
            need_upsample = True
        else:
            feat_S_ds, feat_T_ds = feat_S, feat_T
            H_ds, W_ds = H, W

        B = feat_S_ds.shape[0]
        scale = max(1e-6, float(self.temperaturesa))

        # T->S (Teacher to Student) path is always active
        qt = self.qt_conv(feat_T_ds).reshape(B, -1, H_ds * W_ds).permute(0, 2, 1)
        ks = self.ks_conv(feat_S_ds).reshape(B, -1, H_ds * W_ds)
        vs = self.vs_conv(feat_S_ds).reshape(B, -1, H_ds * W_ds).permute(0, 2, 1)

        att_ts = torch.softmax(torch.bmm(qt, ks) / scale, dim=-1)
        out_ts = torch.bmm(att_ts, vs).permute(0, 2, 1).reshape(B, -1, H_ds, W_ds)

        if activate_s_to_t:
            # S->T (Student to Teacher) path, conditionally activated
            qs = self.qs_conv(feat_S_ds).reshape(B, -1, H_ds * W_ds).permute(0, 2, 1)
            kt = self.kt_conv(feat_T_ds).reshape(B, -1, H_ds * W_ds)
            vt = self.vt_conv(feat_T_ds).reshape(B, -1, H_ds * W_ds).permute(0, 2, 1)

            att_st = torch.softmax(torch.bmm(qs, kt) / scale, dim=-1)
            out_st = torch.bmm(att_st, vt).permute(0, 2, 1).reshape(B, -1, H_ds, W_ds)

            fused = torch.cat([out_st, out_ts], dim=1)
            fused = self.fuse_conv(fused)
        else:
            fused = self.fuse_conv_single(out_ts)


        if need_upsample and (H_ds != H or W_ds != W):
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
        return fused

    def generate_masks(self, fused):
        logits = self.mask_conv(fused)
        B, M, H, W = logits.shape
        masks = F.softmax(logits.reshape(B, M, -1) / max(1e-6, float(self.temperaturesa)), dim=1).reshape(B, M, H, W)
        return masks

    def diversity_loss(self, masks):
        # 鼓励不同掩码关注不重叠区域（简单的余弦相似度正则）
        B, M, H, W = masks.shape
        m_flat = masks.reshape(B, M, -1)  # [B, M, HW]
        m_mean = m_flat.mean(dim=0)       # [M, HW]
        m_norm = F.normalize(m_mean, p=2, dim=1)
        G = m_norm @ m_norm.t()           # [M, M]
        off = G - torch.diag(torch.diag(G))
        return off.sum() / (M * (M - 1) + 1e-12)

    # -------------------- SAMD：频域模块 --------------------
    @staticmethod
    def _riemann_map(real: torch.Tensor, imag: torch.Tensor):
        """
        将复平面映射到单位黎曼球坐标，稳定比较相位/幅度。
        输入/输出形状均与 real/imag 相同（返回 3 个分量列表）。
        """
        deno = (real * real + imag * imag + 1).clamp_min(1e-9)
        xR = (2 * real) / deno
        yR = (2 * imag) / deno
        zR = (real * real + imag * imag - 1) / deno
        return [xR, yR, zR]

    def _get_radial_weight(self, Hf, Wf, device, dtype):
        """
        构造以(0,0)为低频原点的半谱径向半径（r ∈ [0,1]），缓存避免重复构造。
        注意：rFFT2 输出为半谱，横轴长度为 Wf，仍可用归一化网格生成半径。
        """
        key = (Hf, Wf, str(device), str(dtype))
        if key in self._radial_cache:
            return self._radial_cache[key]
        yy = torch.linspace(0, 1, steps=Hf, device=device, dtype=dtype).view(Hf, 1).expand(Hf, Wf)
        xx = torch.linspace(0, 1, steps=Wf, device=device, dtype=dtype).view(1, Wf).expand(Hf, Wf)
        r = torch.sqrt(xx * xx + yy * yy)  # 原点(0,0)处为最低频
        r = r.clamp_max(1.0)
        weight = r.unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
        self._radial_cache[key] = weight
        return weight

    # def _freq_weight_from_context(self, ctx_feat, mask_m):
    #     """
    #     从融合特征 + 掩码得到频带参数：
    #     - beta: 控制带宽（映射到 σ ∈ (0.15, 1.0) 的高斯低通）
    #     - gamma: 低频/高频混合因子（sigmoid 到 (0,1)）
    #     返回：sigma[b,1,1,1], gamma[b,1,1,1]
    #     """
    #     # 掩码化上下文，提升可分辨的局部结构感知
    #     ctx = ctx_feat * mask_m
    #     params = self.freq_head(ctx)  # [B,2,1,1]
    #     beta = torch.sigmoid(params[:, 0:1, ...])  # (0,1)
    #     gamma = torch.sigmoid(params[:, 1:2, ...]) # (0,1)
    #     sigma = 0.15 + 0.85 * beta                 # 避免过窄带
    #     return sigma, gamma

    def _freq_weight_from_context(self, ctx_feat, mask_m):
        # ctx_feat: [B, C, H, W]
        # mask_m:   [B, 1, H, W]

        # 1. 提取感兴趣区域 (ROI)
        roi = ctx_feat * mask_m

        # 计算每个 Mask 的有效面积 (防止除以 0)
        # sum(dim=(2,3)) 表示在 H, W 维度求和
        mask_area = mask_m.sum(dim=(2, 3), keepdim=True).clamp_min(1e-5)  # [B, 1, 1, 1]

        # 2. 计算【掩码区域内】的均值 (Masked Mean) -> 代表语义
        # Sum(Features) / Sum(Mask)
        roi_sum = roi.sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        feat_mean = roi_sum / mask_area

        # 3. 计算【掩码区域内】的标准差 (Masked Std) -> 代表纹理复杂度
        # Std 是区分平滑(天空)和纹理(树)的最强特征
        # var = E[x^2] - (E[x])^2
        roi_sq_sum = (roi ** 2).sum(dim=(2, 3), keepdim=True)
        feat_sq_mean = roi_sq_sum / mask_area
        # 这里的 clamp 也就是为了数值稳定，防止负数开根号
        feat_var = (feat_sq_mean - feat_mean ** 2).clamp_min(1e-5)
        feat_std = torch.sqrt(feat_var)

        # 4. 拼接：均值 + 标准差
        # [B, 2C, 1, 1]
        feat_cat = torch.cat([feat_mean, feat_std], dim=1)

        # 5. 预测参数
        params = self.freq_head(feat_cat)

        beta = torch.sigmoid(params[:, 0:1, ...])
        gamma = torch.sigmoid(params[:, 1:2, ...])
        sigma = 0.15 + 0.85 * beta

        # ================= [再次修改这里] =================
        # 为了可视化区分度，我们采取更激进的策略：
        # 记录每个样本在所有通道中的【最大标准差】(Max Std)
        # B: Batch Size (Sentinel时为1)

        # 1. 展平通道维度 [B, C]
        std_flat = feat_std.view(feat_std.size(0), -1)
        mean_flat = feat_mean.view(feat_mean.size(0), -1)

        # 2. 取最大值 (Max)，而不是均值
        # 这样能捕捉到“最显著的纹理特征”
        representative_std, _ = std_flat.max(dim=1)  # [B]

        # 对于均值，取绝对值的最大值 (因为有正负)
        representative_mean, _ = mean_flat.abs().max(dim=1)  # [B]

        # 调试用：如果您想确认 feat_std 是否真的拉开了差距，可以取消下面的注释
        # if not self.training:
        #     print(f"Mask Mean: {feat_mean.mean().item():.3f}, Mask Std: {feat_std.mean().item():.3f}")

        # [修改] 返回值增加 feat_mean 和 feat_std (取均值变成标量)
        # 注意：这里返回的是 Tensor，带梯度的
        return sigma, gamma, representative_mean, representative_std

    # def _fft_riemann_and_weighted_cka(self, S_feat, T_feat, ctx_feat, mask_m, return_debug=False):
    #     """
    #             针对单个掩码，执行：
    #             1) 通道降维 -> rFFT2 -> 黎曼映射
    #             2) 用 sigma/gamma 生成频带权重（低/高频混合）
    #             3) 加权（按频点）后的 CKA（对三维黎曼坐标平均）
    #             返回：标量 loss
    #             """
    #     """
    #             return_debug=True: 返回 (loss, sigma, gamma, Filter_W, freq_loss_val)
    #             return_debug=False: 返回 (loss, sigma, gamma)
    #             """
    #
    #
    #     # 教师频域分支不回传梯度，降低显存占用
    #     T_feat = T_feat.detach()
    #
    #     # 将 S/T 对齐到共同维度并通道降维以节省显存
    #     S_m = self.spectral_proj(S_feat)
    #     T_m = self.spectral_proj(T_feat)
    #
    #     # 掩码增强：学生乘 (1+mask)，教师乘 mask，鼓励学生在掩码区域“追老师”
    #     S_m = S_m * (1.0 + mask_m)
    #     T_m = T_m * mask_m
    #
    #     # # 下采样（频域计算分辨率不宜过大）
    #     # S_ds, (Hs, Ws), _ = self._maybe_downsample(S_m)
    #     # T_ds, _, _ = self._maybe_downsample(T_m)
    #
    #     # rFFT2：半谱，显存更低；需要同一分辨率
    #     F_S = torch.fft.rfft2(S_m, dim=(2, 3), norm='ortho')  # [B,C,Hf,Wf']
    #     F_T = torch.fft.rfft2(T_m, dim=(2, 3), norm='ortho')
    #     real_S, imag_S = F_S.real, F_S.imag
    #     real_T, imag_T = F_T.real, F_T.imag
    #
    #     # 黎曼球映射，得到三分量
    #     R_S = self._riemann_map(real_S, imag_S)  # list of 3 tensors
    #     R_T = self._riemann_map(real_T, imag_T)
    #
    #     # 基于上下文 + 掩码，得到频带参数并生成权重图
    #     sigma, gamma = self._freq_weight_from_context(ctx_feat, mask_m)  # [B,1,1,1]
    #
    #     B, C, Hf, Wf = real_S.shape
    #     radial = self._get_radial_weight(Hf, Wf, device=real_S.device, dtype=real_S.dtype)  # [1,1,Hf,Wf]
    #     # 低通（高斯）与高通（互补）
    #     w_low = torch.exp(-(radial * radial) / (2.0 * (sigma ** 2)))   # [B,1,Hf,Wf]（广播）
    #     w_high = 1.0 - w_low
    #     W = gamma * w_high + (1.0 - gamma) * w_low   # [B,1,Hf,Wf]
    #     sqrtW = torch.sqrt(W.clamp_min(1e-8))
    #
    #     # 对每个黎曼分量进行加权，随后展平到 [B, D] 用 CKA
    #     losses = []
    #     for s_comp, t_comp in zip(R_S, R_T):
    #         # [B,C,Hf,Wf] * [B,1,Hf,Wf] -> [B,C,Hf,Wf]
    #         s_w = (s_comp * sqrtW).reshape(B, -1)
    #         t_w = (t_comp * sqrtW).reshape(B, -1)
    #         losses.append(self.linear_CKA_loss(s_w, t_w))
    #     loss = sum(losses) / 3.0
    #
    #     # 及时释放中间变量，降低峰值显存
    #     del F_S, F_T, real_S, imag_S, real_T, imag_T, R_S, R_T, w_low, w_high, W, sqrtW
    #
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #
    #     if return_debug:
    #         # 返回: Loss, Sigma, Gamma, FilterMap(W)
    #         return loss, sigma, gamma, W
    #     else:
    #         return loss, sigma, gamma
    #
    #     # return loss, sigma, gamma

    def _fft_riemann_and_weighted_cka(self, S_feat, T_feat, ctx_feat, mask_m, return_debug=False):
        """
        针对单个掩码，执行：
        1) 通道降维 -> rFFT2 -> 黎曼映射
        2) 用 sigma/gamma 生成频带权重（低/高频混合）
        3) 加权（按频点）后的 CKA（对三维黎曼坐标平均）
        返回：标量 loss
        """

        # 教师频域分支不回传梯度，降低显存占用
        T_feat = T_feat.detach()

        # 将 S/T 对齐到共同维度并通道降维以节省显存
        S_m = self.spectral_proj(S_feat)
        T_m = self.spectral_proj(T_feat)

        # 掩码增强：学生乘 (1+mask)，教师乘 mask，鼓励学生在掩码区域“追老师”
        S_m = S_m * (1.0 + mask_m)
        T_m = T_m * mask_m

        # rFFT2：半谱，显存更低；需要同一分辨率
        F_S = torch.fft.rfft2(S_m, dim=(2, 3), norm='ortho')  # [B,C,Hf,Wf']
        F_T = torch.fft.rfft2(T_m, dim=(2, 3), norm='ortho')
        real_S, imag_S = F_S.real, F_S.imag
        real_T, imag_T = F_T.real, F_T.imag

        # 黎曼球映射，得到三分量
        R_S = self._riemann_map(real_S, imag_S)  # list of 3 tensors
        R_T = self._riemann_map(real_T, imag_T)

        # 基于上下文 + 掩码，得到频带参数并生成权重图
        #sigma, gamma = self._freq_weight_from_context(ctx_feat, mask_m)  # [B,1,1,1]
        sigma, gamma, f_mean, f_std = self._freq_weight_from_context(ctx_feat, mask_m)

        B, C, Hf, Wf = real_S.shape
        radial = self._get_radial_weight(Hf, Wf, device=real_S.device, dtype=real_S.dtype)  # [1,1,Hf,Wf]

        # 低通（高斯）与高通（互补）
        w_low = torch.exp(-(radial * radial) / (2.0 * (sigma ** 2)))  # [B,1,Hf,Wf]（广播）
        w_high = 1.0 - w_low
        W = gamma * w_high + (1.0 - gamma) * w_low  # [B,1,Hf,Wf]
        sqrtW = torch.sqrt(W.clamp_min(1e-8))

        # ==================== [核心修改开始] ====================
        losses = []

        # 判断当前是 正常训练(B>1) 还是 可视化监测(B=1)
        if B == 1:
            # [新增逻辑] 针对单张图片的特殊处理
            # 这里的 loss 是为了画图展示 "距离在缩小"，MSE 是最佳替代品
            for s_comp, t_comp in zip(R_S, R_T):
                # 直接计算加权后的特征距离
                s_w = s_comp * sqrtW
                t_w = t_comp * sqrtW
                losses.append(self.mse(s_w, t_w))
            loss = sum(losses)
        else:
            # [原有逻辑] 正常的 CKA 计算（用于训练，需要 Batch 统计特性）
            for s_comp, t_comp in zip(R_S, R_T):
                # [B,C,Hf,Wf] * [B,1,Hf,Wf] -> [B,C,Hf,Wf]
                s_w = (s_comp * sqrtW).reshape(B, -1)
                t_w = (t_comp * sqrtW).reshape(B, -1)
                losses.append(self.linear_CKA_loss(s_w, t_w))
            loss = sum(losses) / 3.0
        # ==================== [核心修改结束] ====================

        # 及时释放中间变量，降低峰值显存
        # 注意：这里千万不要 del W，因为下面 return_debug 要用到它
        del F_S, F_T, real_S, imag_S, real_T, imag_T, R_S, R_T, w_low, w_high, sqrtW

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_debug:
            # 返回: Loss, Sigma, Gamma, FilterMap(W)
            return loss, sigma, gamma, W, f_mean, f_std
        else:
            return loss, sigma, gamma


    # -------------------- 前向 --------------------
    def forward(self, feat_S, feat_T, feature_transform="CAM_CKA", num_masks=None, activate_s_to_t=True, visualization_mode=False):
        if num_masks is not None:
            self.num_masks = int(num_masks)

        B, Cs, H, W = feat_S.shape
        _, Ct, _, _ = feat_T.shape
        # if not self._initialized:
        #     self._lazy_init(Cs, Ct)

        # 教师侧不回传梯度，避免保存教师计算图
        feat_T = feat_T.detach()

        # 兼容旧模式（保持不变）
        if feature_transform in ["CAM_CKA", "CAM_MSE", "PAM_CKA", "PAM_MSE",
                                 "gridPAM_CKA", "gridPAM_MSE",
                                 "separately_CAMgridPAM_CKA", "separately_CAMgridPAM_MSE"]:
            # ...... 保持原实现（略） ......
            # 为简洁起见，此处应复用你现有代码的对应分支。
            S_mapped = self.align_s_conv(feat_S)
            T_mapped = self.align_t_conv(feat_T)
            if feature_transform == "CAM_CKA":
                CAM_S = self.CAM(S_mapped).reshape(B, -1)
                CAM_T = self.CAM(T_mapped).reshape(B, -1)
                return self.linear_CKA_loss(CAM_S, CAM_T)
            if feature_transform == "CAM_MSE":
                return self.mse(self.CAM(S_mapped), self.CAM(T_mapped))
            if feature_transform == "PAM_CKA":
                PAM_S = self.PAM(S_mapped).reshape(B, -1)
                PAM_T = self.PAM(T_mapped).reshape(B, -1)
                return self.linear_CKA_loss(PAM_S, PAM_T)
            if feature_transform == "PAM_MSE":
                return self.mse(self.PAM(S_mapped), self.PAM(T_mapped))
            if feature_transform in ["gridPAM_CKA", "gridPAM_MSE"]:
                firstpart_S = torch.chunk(S_mapped, 5, dim=2)
                partsS = []
                for i in firstpart_S:
                    secondpart_S = torch.chunk(i, 5, dim=3)
                    for j in secondpart_S:
                        partsS.append(self.PAM(j))
                firstpart_T = torch.chunk(T_mapped, 5, dim=2)
                partsT = []
                for i in firstpart_T:
                    secondpart_T = torch.chunk(i, 5, dim=3)
                    for j in secondpart_T:
                        partsT.append(self.PAM(j))
                loss = 0
                n = len(partsS)
                for i in range(n):
                    if feature_transform == "gridPAM_CKA":
                        S_p = partsS[i].reshape(partsS[i].size(0), -1)
                        T_p = partsT[i].reshape(partsT[i].size(0), -1)
                        loss += self.linear_CKA_loss(S_p, T_p)
                    else:
                        loss += self.mse(partsS[i], partsT[i])
                return loss / max(1, n)
            if feature_transform in ["separately_CAMgridPAM_CKA", "separately_CAMgridPAM_MSE"]:
                CAM_S = self.CAM(S_mapped)
                CAM_T = self.CAM(T_mapped)
                firstpart_S = torch.chunk(S_mapped, 5, dim=2)
                partsS = []
                for i in firstpart_S:
                    secondpart_S = torch.chunk(i, 5, dim=3)
                    for j in secondpart_S:
                        partsS.append(self.PAM(j))
                firstpart_T = torch.chunk(T_mapped, 5, dim=2)
                partsT = []
                for i in firstpart_T:
                    secondpart_T = torch.chunk(i, 5, dim=3)
                    for j in secondpart_T:
                        partsT.append(self.PAM(j))
                loss_P = 0
                n = len(partsS)
                for i in range(n):
                    if feature_transform == "separately_CAMgridPAM_CKA":
                        S_p = partsS[i].reshape(partsS[i].size(0), -1)
                        T_p = partsT[i].reshape(partsT[i].size(0), -1)
                        loss_P += self.linear_CKA_loss(S_p, T_p)
                    else:
                        loss_P += self.mse(partsS[i], partsT[i])
                loss_P = loss_P / max(1, n)
                if feature_transform == "separately_CAMgridPAM_CKA":
                    return self.linear_CKA_loss(CAM_S.reshape(B, -1), CAM_T.reshape(B, -1)), loss_P
                else:
                    return self.mse(CAM_S, CAM_T), loss_P

        # -------------------- 新模式：STCA_ASCM_M_CKA -> SAMD 频域多掩码对齐 --------------------
        if feature_transform in ["STCA_ASCM_M_CKA", "STCA_ASCM_M_MSE"]:
            # 1) STCA 融合得到上下文
            fused = self.stca_ff(feat_S, feat_T, activate_s_to_t=activate_s_to_t)                      # [B,F,H,W]
            masks = self.generate_masks(fused)                        # [B,M,H,W]
            S_al = self.align_s_conv(feat_S)                          # [B,F,H,W]
            T_al = self.align_t_conv(feat_T)                          # [B,F,H,W]

            # # ================= [可视化模式] =================
            # # 如果开启，直接返回第一个 mask 的详细信息用于绘图
            # if visualization_mode:
            #     # mask_m = masks[:, 0:1, :, :]  # 取第1个mask做演示
            #     # freq_loss, s_val, g_val, W_map, f_mean, f_std = self._fft_riemann_and_weighted_cka(
            #     #     S_feat=S_al, T_feat=T_al, ctx_feat=fused, mask_m=mask_m, return_debug=False
            #     # )
            #     # # 返回:
            #     # return mask_m, W_map, s_val, g_val, freq_loss, f_mean, f_std
            #     return masks

            # ================= [正常训练模式] =================
            per_mask_losses = []
            M = masks.shape[1]

            total_sigma = 0
            total_gamma = 0

            mask_details = []  # [新增] 用于存储每个 mask 的详细统计数据

            for m in range(M):
                mask_m = masks[:, m:m+1, :, :]                        # [B,1,H,W]

                # 空间分支：CAM + PAM
                S_masked = S_al * (1.0 + mask_m)
                T_masked = T_al * mask_m

                cam_s = self.CAM(S_masked).reshape(B, -1)
                pam_s = self.PAM(S_masked).reshape(B, -1)


                # with torch.no_grad():
                #     cam_t = self.CAM(T_masked).reshape(B, -1)
                #     pam_t = self.PAM(T_masked).reshape(B, -1)
                #
                # if self.align_method == 'cka' or feature_transform.endswith("_CKA"):
                #     loss_cam = self.linear_CKA_loss(cam_s, cam_t)
                #     loss_pam = self.linear_CKA_loss(pam_s, pam_t)
                #     spatial_loss = 0.5 * (loss_cam + loss_pam)
                # else:
                #     spatial_loss = 0.5 * (self.mse(cam_s, cam_t) + self.mse(pam_s, pam_t))

                # 只有在非可视化模式下才计算空间 Loss 以节省显存
                spatial_loss = torch.tensor(0.0).to(feat_S.device)
                if not visualization_mode:
                    cam_s = self.CAM(S_masked).reshape(B, -1)
                    pam_s = self.PAM(S_masked).reshape(B, -1)
                    with torch.no_grad():
                        cam_t = self.CAM(T_masked).reshape(B, -1)
                        pam_t = self.PAM(T_masked).reshape(B, -1)
                    spatial_loss = 0.5 * (self.linear_CKA_loss(cam_s, cam_t) + self.linear_CKA_loss(pam_s, pam_t))

                # 频域分支：SAMD
                # SAMD：在频域内做按掩码自适应的加权 CKA
                # freq_loss = self._fft_riemann_and_weighted_cka(
                #     S_feat=S_al, T_feat=T_al, ctx_feat=fused, mask_m=mask_m
                # )

                # 频域分支计算：始终执行以获取统计量 f_mean, f_std
                # 修改 _fft_riemann_and_weighted_cka 的调用，获取返回值
                freq_loss, sigma_val, gamma_val, W_map, f_mean, f_std = self._fft_riemann_and_weighted_cka(
                    S_feat=S_al, T_feat=T_al, ctx_feat=fused, mask_m=mask_m, return_debug=True
                )

                # 记录均值供 Tensorboard 使用
                # total_sigma += sigma_val.mean()
                # total_gamma += gamma_val.mean()

                # 融合（默认等权，可按需调整 lambda_freq）
                loss_m = (1.0 - self.lambda_freq) * spatial_loss + self.lambda_freq * freq_loss
                per_mask_losses.append(loss_m)

                # [核心新增]：记录当前 Mask 的统计值 (转为 float)
                # 记录详情（这就是你要存入 CSV 的“放大版”数据）
                mask_details.append({
                    f'mask_{m}_mean': f_mean.mean().item(),  # 这是你代码里的 Representative Mean (Max 版)
                    f'mask_{m}_std': f_std.mean().item(),  # 这是你代码里的 Representative Std (Max 版)
                    f'mask_{m}_sigma': sigma_val.mean().item(),
                    f'mask_{m}_gamma': gamma_val.mean().item(),
                    f'mask_{m}_loss': freq_loss.item()
                })


                loss_m = (1.0 - self.lambda_freq) * spatial_loss + self.lambda_freq * freq_loss
                per_mask_losses.append(loss_m)


            # ================= [关键返回值修改] =================
            if visualization_mode:
                # 返回：掩码，统计详情 (给 CSV 用)，和第一个 Filter 图 (给画图用)
                return masks, mask_details, W_map


            align_loss = torch.stack(per_mask_losses).mean()
            div_loss = self.diversity_loss(masks)
            total = align_loss + self.diversity_weight * div_loss

            # [修改返回值]：返回总 Loss 和 详细记录列表
            return total, mask_details

        # 兜底：默认 CAM_CKA
        CAM_S = self.CAM(self.align_s_conv(feat_S)).reshape(B, -1)
        CAM_T = self.CAM(self.align_t_conv(feat_T)).reshape(B, -1)
        return self.linear_CKA_loss(CAM_S, CAM_T)










