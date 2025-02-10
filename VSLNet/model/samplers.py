import math

import numpy as np
import torch
import torch.nn as nn
from model.utils import TopKOperator

MIN_EPSILON = np.finfo(np.float32).min


class BaseSampler(nn.Module):
    def __init__(self, efficiency, start_idx, end_idx, *args, **kwargs):
        super().__init__()
        self.efficiency = efficiency
        self.start_idx = start_idx
        self.end_idx = end_idx
        assert self.start_idx == 0

    def forward(self, x, *args, v_mask=None, **kwargs):
        # x - (B, L, F)
        x_samp, x_mask = self.sample_features(x, v_mask)
        return x_samp, x_mask

    def sample_features(self, x, v_mask):
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs):
        return 0.0

    def compute_sampling_rate(self, predicted_masks, video_masks):
        """
        Args:
            predicted_masks - (B, L)
            video_masks - (B, L)
        """
        # filter out invalid predictions
        predicted_masks = predicted_masks * video_masks
        # Count sampling rate as number of frames sampled / total number of frames
        sample_count = predicted_masks.sum(dim=1)  # (B, )
        total_count = video_masks.sum(dim=1)  # (B, )
        sampling_rate = sample_count / (total_count + 1e-12)
        return sampling_rate

    @classmethod
    def from_configs(cls, configs):
        return cls(
            configs.feature_sampler_efficiency,
            configs.feature_mask_idxs[0],
            configs.feature_mask_idxs[1],
        )


class RandomSampler(BaseSampler):
    def sample_features(self, x, v_mask):
        """
        Args:
            x - (B, L, F)
            v_mask - (B, L)
        """
        B, _, _ = x.shape
        feat_masks = torch.zeros_like(x)
        feat_masks[:, :, self.end_idx + 1 :] = 1
        for i in range(B):
            L = int(v_mask[i].sum().item())
            n2samp = int((1 - self.efficiency) * L)
            idxs2samp = np.random.permutation(L)[:n2samp]
            feat_masks[i, idxs2samp, self.start_idx : self.end_idx + 1] = 1
        x = x * feat_masks
        return x, feat_masks[:, :, self.start_idx]


class UniformSampler(BaseSampler):
    def sample_features(self, x, v_mask):
        """
        Args:
            x - (B, L, F)
            v_mask - (B, L)
        """
        B, _, _ = x.shape
        feat_masks = torch.zeros_like(x)
        feat_masks[:, :, self.end_idx + 1 :] = 1
        for i in range(B):
            L = int(v_mask[i].sum().item())
            n2samp = int((1 - self.efficiency) * L)
            if n2samp > 0:
                delta = L // float(n2samp)
                start = int(np.random.uniform(0.0, delta / 2.0))
                idxs2samp = np.arange(start, L - 1, delta)
                feat_masks[i, idxs2samp, self.start_idx : self.end_idx + 1] = 1
        x = x * feat_masks
        return x, feat_masks[:, :, self.start_idx]


class AllSampler(BaseSampler):
    def sample_features(self, x, v_mask):
        """
        Args:
            x - (B, L, F)
            v_mask - (B, L)
        """
        feat_masks = torch.ones_like(x) * v_mask.unsqueeze(2)
        x = x * feat_masks
        return x, feat_masks[:, :, self.start_idx]


class ZeroSampler(BaseSampler):
    def sample_features(self, x, v_mask):
        """
        Args:
            x - (B, L, F)
            v_mask - (B, L)
        """
        feat_masks = torch.ones_like(x) * v_mask.unsqueeze(2)
        feat_masks[:, :, self.start_idx : self.end_idx + 1] = 0
        x = x * feat_masks
        return x, feat_masks[:, :, self.start_idx]


class TransformerV1Sampler(BaseSampler):
    def __init__(
        self,
        efficiency,
        start_idx,
        end_idx,
        dim,
        niters,
        loss_type,
        disable_stepwise_loss,
        mask_prev,
        use_video_mask_for_loss,
    ):
        super().__init__(efficiency, start_idx, end_idx)
        self.selection_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
        )
        assert self.start_idx == 0
        self.niters = niters
        self.loss_type = loss_type
        self.disable_stepwise_loss = disable_stepwise_loss
        self.mask_prev = mask_prev
        self.use_video_mask_for_loss = use_video_mask_for_loss
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Linear)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def forward(
        self,
        video_features,
        query_features=None,
        v_mask=None,
        q_mask=None,
        video_affine=None,
        feature_encoder=None,
        cq_attention=None,
        cq_concat=None,
        gs_tau=None,
        niters=None,
    ):
        # video_features - (B, L, F)
        # v_mask - (B, L)
        B, L, F = video_features.shape

        fine_features = video_features[:, :, self.start_idx : self.end_idx + 1]
        coarse_features = video_features[:, :, self.end_idx + 1 :]
        # one-time computation
        query_features = feature_encoder(query_features, mask=q_mask)
        # Compute scores iteratively
        predicted_masks = []
        niters = self.niters if niters is None else niters
        all_mask = torch.zeros_like(v_mask)
        for i in range(niters):
            video_feats_i = torch.cat(
                [fine_features * all_mask.unsqueeze(2), coarse_features], dim=2
            )
            video_feats_i = video_affine(video_feats_i)
            video_feats_i = feature_encoder(video_feats_i, mask=v_mask)
            features_i = cq_attention(video_feats_i, query_features, v_mask, q_mask)
            features_i = cq_concat(features_i, query_features, q_mask)  # (B, L, F)
            selection_logits = self.selection_mlp(features_i)  # (B, L, 2)
            if self.mask_prev:
                # Ignore previously selected locations
                to_mask = all_mask.detach() == 1
                to_mask = torch.stack([to_mask, torch.zeros_like(to_mask)], dim=2)
                to_mask = to_mask.float()
                selection_logits = (
                    1 - to_mask
                ) * selection_logits + to_mask * MIN_EPSILON
            selection_oh = nn.functional.gumbel_softmax(
                selection_logits, tau=gs_tau, hard=True, dim=2
            )  # (B, L, 2)
            per_step_mask = selection_oh[..., 0]
            # Ignore invalid mask
            all_mask = all_mask + per_step_mask * v_mask
            if not self.mask_prev:
                all_mask = torch.clamp(all_mask, 0.0, 1.0)
            predicted_masks.append(per_step_mask * v_mask)

        sampled_video_features = torch.cat(
            [fine_features * all_mask.unsqueeze(2), coarse_features], dim=2
        )
        predicted_masks = torch.stack(predicted_masks, dim=1)  # (B, niters, L)
        return sampled_video_features, predicted_masks

    def compute_loss(self, predicted_masks, video_mask, **kwargs):
        if self.use_video_mask_for_loss:
            return self.compute_loss_with_video_mask(predicted_masks, video_mask)
        else:
            return self.compute_loss_without_video_mask(predicted_masks)

    def compute_loss_with_video_mask(self, predicted_masks, video_masks):
        # predicted_masks - (B, niters, L)
        # video_masks - (B, L)
        _, niters, L = predicted_masks.shape
        assert niters <= self.niters

        # compute lengths per video
        lengths = video_masks.sum(dim=1)  # (B, )

        # estimate budgets
        frac_total = 1 - self.efficiency
        frac_per_step = frac_total / self.niters
        frac_niter = frac_per_step * niters  # (in case niters is less then self.niters)

        # estimate loss for complete selection
        total_mask = predicted_masks.sum(dim=1)  # (B, L)
        if not self.mask_prev:
            total_mask = torch.clamp(total_mask, 0.0, 1.0)
        total_loss = self.loss_util_w_video_mask(
            total_mask,
            lengths,
            frac_niter,
        )

        # estimate loss for each selection step
        per_step_loss = 0.0
        for i in range(niters):
            step_mask = predicted_masks[:, i]
            step_loss = self.loss_util_w_video_mask(step_mask, lengths, frac_per_step)
            per_step_loss = per_step_loss + step_loss

        # average loss
        total_loss = (total_loss + per_step_loss) / (niters + 1)
        return total_loss

    def loss_util_w_video_mask(self, mask, lengths, required_frac):
        """
        mask - (B, L) selections
        lengths - (B, ) true lengths of each video after ignoring masked locs
        required_frac - GT scalar fraction of video samples to select
        """
        n_selected = mask.sum(dim=1)  # (B, )
        selected_frac = n_selected / lengths
        if self.loss_type == "sample":
            loss = (selected_frac - required_frac).pow(2).mean()  # (B, )
        elif self.loss_type == "batch":
            loss = (selected_frac.mean() - required_frac).pow(2)
        return loss

    def compute_loss_without_video_mask(self, predicted_masks):
        # predicted_masks - (B, niters, L)
        _, niters, L = predicted_masks.shape
        assert niters <= self.niters
        nsamples_total = (1 - self.efficiency) * L
        nsamples_per_step = max(1.0, nsamples_total / self.niters)
        total_mask = predicted_masks.sum(dim=1)
        if not self.mask_prev:
            total_mask = torch.clamp(total_mask, 0.0, 1.0)
        total_loss = self.loss_util(total_mask, nsamples_per_step * niters)
        if not self.disable_stepwise_loss:
            per_step_loss = self.compute_stepwise_loss(
                predicted_masks, nsamples_per_step
            )
            total_loss = (total_loss + per_step_loss) / (niters + 1)
        return total_loss

    def compute_stepwise_loss(self, predicted_masks, nsamples_per_step):
        _, niters, _ = predicted_masks.shape
        per_step_loss = 0.0
        for i in range(niters):
            step_mask = predicted_masks[:, i]
            step_loss = self.loss_util(step_mask, nsamples_per_step)
            per_step_loss = per_step_loss + step_loss
        return per_step_loss

    def loss_util(self, mask, nsamples):
        # mask - (B, L)
        _, L = mask.shape
        if self.loss_type == "sample":
            loss = ((mask.mean(dim=1) - nsamples / L) ** 2).mean()
        elif self.loss_type == "batch":
            loss = (mask.mean() - nsamples / L) ** 2
        return loss

    def compute_sampling_rate(self, predicted_masks, video_masks):
        """
        Args:
            predicted_masks - (B, niters, L)
            video_masks - (B, L)
        """
        # Sum predicted masks to get total mask
        total_masks = torch.clamp(predicted_masks.sum(dim=1), 0.0, 1.0)
        sampling_rate = super().compute_sampling_rate(total_masks, video_masks)
        return sampling_rate

    @classmethod
    def from_configs(cls, configs):
        return cls(
            configs.feature_sampler_efficiency,
            configs.feature_mask_idxs[0],
            configs.feature_mask_idxs[1],
            configs.dim,
            configs.sampler_niters,
            configs.sampler_loss_type,
            configs.disable_stepwise_loss,
            configs.sampler_mask_prev,
            configs.video_mask_for_sampler_loss,
        )

    def sample_random_niters(self):
        return np.random.randint(1, self.niters + 1)

    def calculate_niters_for_efficiency(self, efficiency):
        samples_per_step = (1 - self.efficiency) / self.niters
        reqd_samples = 1 - efficiency
        niters = int(reqd_samples / samples_per_step)
        return niters


class LiteEvalSampler(BaseSampler):
    def __init__(
        self,
        efficiency,
        start_idx,
        end_idx,
        v_dim,
        q_dim,
        coarse_hdim,
        fine_hdim,
        loss_type,
        use_video_mask_for_loss,
    ):
        super().__init__(efficiency, start_idx, end_idx)
        assert self.start_idx == 0
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.coarse_hdim = coarse_hdim
        self.fine_hdim = fine_hdim
        self.fine_idim = end_idx - start_idx + 1
        self.coarse_idim = v_dim - self.fine_idim
        self.loss_type = loss_type
        self.use_video_mask_for_loss = use_video_mask_for_loss

        self.coarse_lstm = nn.LSTMCell(self.coarse_idim + q_dim, coarse_hdim)
        self.fine_lstm = nn.LSTMCell(
            self.coarse_idim + self.fine_idim + q_dim, fine_hdim
        )
        self.selection_mlp = nn.Linear(
            self.coarse_idim + q_dim + 2 * fine_hdim, 2, bias=False
        )

        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Linear)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def forward(
        self,
        video_features,
        query_features=None,
        v_mask=None,
        q_mask=None,
        gs_tau=None,
        **kwargs,
    ):
        # video_features - (B, Lv, Fv)
        # query_features - (B, Lq, Fq)
        # v_mask - (B, Lv)
        # q_mask - (B, Lq)
        B, Lv, F = video_features.shape
        device = video_features.device

        fine_features = video_features[:, :, self.start_idx : self.end_idx + 1]
        coarse_features = video_features[:, :, self.end_idx + 1 :]
        # average query feature
        q_feat = (query_features * q_mask.unsqueeze(2)).sum(dim=1) / (
            q_mask.sum(dim=1, keepdim=True) + 1e-10
        )  # (B, Fq)

        # initialize hidden states
        fine_hs = torch.zeros(B, self.fine_hdim).to(device)
        fine_cs = torch.zeros(B, self.fine_hdim).to(device)
        coarse_hs = torch.zeros(B, self.coarse_hdim).to(device)
        coarse_cs = torch.zeros(B, self.coarse_hdim).to(device)

        # iteratively pick features
        predicted_masks = []
        for t in range(Lv):
            fine_t = fine_features[:, t]  # (B, Ff)
            coarse_t = coarse_features[:, t]  # (B, Fc)
            # coarse-lstm update
            coarse_hs, coarse_cs = self.coarse_lstm(
                torch.cat([coarse_t, q_feat], dim=1), (coarse_hs, coarse_cs)
            )
            # conditional gating
            selection_logits = self.selection_mlp(
                torch.cat([coarse_t, q_feat, fine_hs, fine_cs], dim=1)
            )  # (B, 2)
            selection_oh = nn.functional.gumbel_softmax(
                selection_logits, tau=gs_tau, hard=True, dim=1
            )  # (B, 2)
            b_t = selection_oh[:, 0]  # (B, )
            b_t = (b_t * v_mask[:, t]).unsqueeze(1)  # ignore invalid video features
            # fine-lstm update
            ## compute actual estimated
            fine_hs_bar, fine_cs_bar = self.fine_lstm(
                torch.cat([coarse_t, fine_t, q_feat], dim=1), (fine_hs, fine_cs)
            )
            ## compute ignored estimate
            fine_hs_ = torch.cat([coarse_hs, fine_hs[:, self.coarse_hdim :]], dim=1)
            fine_cs_ = torch.cat([coarse_cs, fine_cs[:, self.coarse_hdim :]], dim=1)
            ## gated estimate
            fine_hs = fine_hs_bar * b_t + (1 - b_t) * fine_hs_
            fine_cs = fine_cs_bar * b_t + (1 - b_t) * fine_cs_
            # update mask
            predicted_masks.append(b_t)

        predicted_masks = torch.cat(predicted_masks, dim=1)  # (B, Lv)
        sampled_video_features = torch.cat(
            [fine_features * predicted_masks.unsqueeze(2), coarse_features], dim=2
        )
        return sampled_video_features, predicted_masks

    def compute_loss(self, predicted_masks, video_mask, **kwargs):
        if self.use_video_mask_for_loss:
            return self.compute_loss_with_video_mask(predicted_masks, video_mask)
        else:
            return self.compute_loss_without_video_mask(predicted_masks)

    def compute_loss_with_video_mask(self, predicted_masks, video_masks):
        # predicted_masks - (B, L)
        # video_masks - (B, L)
        B, L = predicted_masks.shape

        # compute lengths per video
        lengths = video_masks.sum(dim=1)  # (B, )

        # estimate budgets
        frac_total = 1 - self.efficiency

        # estimate loss
        loss = self.loss_util_w_video_mask(predicted_masks, lengths, frac_total)

        return loss

    def loss_util_w_video_mask(self, mask, lengths, required_frac):
        """
        mask - (B, L) selections
        lengths - (B, ) true lengths of each video after ignoring masked locs
        required_frac - GT scalar fraction of video samples to select
        """
        n_selected = mask.sum(dim=1)  # (B, )
        selected_frac = n_selected / lengths
        if self.loss_type == "sample":
            loss = (selected_frac - required_frac).pow(2).mean()  # (B, )
        elif self.loss_type == "batch":
            loss = (selected_frac.mean() - required_frac).pow(2)
        return loss

    def compute_loss_without_video_mask(self, predicted_masks):
        # predicted_masks - (B, L)
        _, L = predicted_masks.shape
        nsamples = (1 - self.efficiency) * L
        total_loss = self.loss_util(predicted_masks, nsamples)
        return total_loss

    def loss_util(self, mask, nsamples):
        # mask - (B, L)
        _, L = mask.shape
        if self.loss_type == "sample":
            loss = ((mask.mean(dim=1) - nsamples / L) ** 2).mean()
        elif self.loss_type == "batch":
            loss = (mask.mean() - nsamples / L) ** 2
        return loss

    @classmethod
    def from_configs(cls, configs):
        return cls(
            configs.feature_sampler_efficiency,
            configs.feature_mask_idxs[0],
            configs.feature_mask_idxs[1],
            configs.video_feature_dim,
            configs.dim,
            configs.sampler_coarse_hdim,
            configs.sampler_fine_hdim,
            configs.sampler_loss_type,
            configs.video_mask_for_sampler_loss,
        )


class OCSampler(BaseSampler):
    def __init__(
        self,
        efficiency,
        start_idx,
        end_idx,
        v_dim,
        q_dim,
        max_len,
    ):
        super().__init__(efficiency, start_idx, end_idx)
        self.max_len = max_len
        assert self.start_idx == 0
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.fine_idim = end_idx - start_idx + 1
        self.coarse_idim = v_dim - self.fine_idim
        self.projector = nn.Sequential(
            nn.Linear(self.coarse_idim + q_dim, self.coarse_idim),
            nn.ReLU(),
            nn.Linear(self.coarse_idim, 1),
        )
        self.topk_operator = TopKOperator(hard=True)
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Linear)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def forward(
        self,
        video_features,
        query_features=None,
        v_mask=None,
        q_mask=None,
        gs_tau=None,
        *args,
        **kwargs,
    ):
        # video_features - (B, Lv, Fv)
        # query_features - (B, Lq, Fq)
        # v_mask - (B, Lv)
        # q_mask - (B, Lq)
        B, L, F = video_features.shape
        device = video_features.device

        fine_features = video_features[:, :, self.start_idx : self.end_idx + 1]
        coarse_features = video_features[:, :, self.end_idx + 1 :]
        # average query feature
        q_feat = (query_features * q_mask.unsqueeze(2)).sum(dim=1) / (
            q_mask.sum(dim=1, keepdim=True) + 1e-10
        )  # (B, Fq)
        q_feat = q_feat.unsqueeze(1).expand(-1, L, -1)
        # compute scores
        inputs = torch.cat([coarse_features, q_feat], dim=2)
        scores = self.projector(inputs).squeeze(2)  # (B, L)
        topk_mask = []
        video_lengths = v_mask.sum(dim=1)
        for b in range(B):
            k = int(math.floor((1 - self.efficiency) * video_lengths[b].item()))
            if k == 0:
                topk_mask.append(torch.zeros(1, L).to(device))
            else:
                topk_mask.append(
                    self.topk_operator(scores[b : b + 1], k, gs_tau)  # (B, 1)
                )
        topk_mask = torch.cat(topk_mask, dim=0)  # (B, L)
        # ignore invalid features
        topk_mask = topk_mask * v_mask
        sampled_video_features = torch.cat(
            [fine_features * topk_mask.unsqueeze(2), coarse_features], dim=2
        )
        return sampled_video_features, topk_mask

    @classmethod
    def from_configs(cls, configs):
        return cls(
            configs.feature_sampler_efficiency,
            configs.feature_mask_idxs[0],
            configs.feature_mask_idxs[1],
            configs.video_feature_dim,
            configs.dim,
            configs.max_pos_len,
        )
