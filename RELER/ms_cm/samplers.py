import math

import numpy as np
import torch
import torch.nn as nn

from utils.topk_utils import TopKOperator

MIN_EPSILON = np.finfo(np.float32).min


class BaseSampler(nn.Module):
    def __init__(self, efficiency, start_idx, end_idx, *args, **kwargs):
        super().__init__()
        self.efficiency = efficiency
        self.start_idx = start_idx
        self.end_idx = end_idx
        assert self.start_idx == 0

    def forward(self, x, src_vid_mask_total=None, cache=None, *args, **kwargs):
        # x - list of (B, L/N, F) tensors
        split_len = x[0].shape[1]
        x = torch.cat(x, dim=1)
        src_vid_mask_total = torch.cat(src_vid_mask_total, dim=1)
        if cache:
            x_mask = cache["sampler_masks"]
            x_samp = x * x_mask
        else:
            x_samp, x_mask = self.sample_features(x, src_vid_mask_total)
        x_samp = torch.split(x_samp, split_len, dim=1)
        return x_samp, x_mask

    def sample_features(self, x, v_mask, cache):
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs):
        return 0.0

    def compute_sampling_rate(self, predicted_masks, video_masks):
        """
        Args:
            predicted_masks - (B, L, F)
            video_masks - (B, L)
        """
        # filter out invalid predictions
        predicted_masks = predicted_masks[:, :, self.start_idx] * video_masks
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
        return x, feat_masks


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
            delta = L // float(n2samp)
            start = int(np.random.uniform(0.0, delta / 2.0))
            idxs2samp = np.arange(start, L - 1, delta)
            feat_masks[i, idxs2samp, self.start_idx : self.end_idx + 1] = 1
        x = x * feat_masks
        return x, feat_masks


class AllSampler(BaseSampler):
    def sample_features(self, x, v_mask):
        """
        Args:
            x - (B, L, F)
            v_mask - (B, L)
        """
        feat_masks = torch.ones_like(x) * v_mask.unsqueeze(2)
        x = x * feat_masks
        return x, feat_masks


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
        return x, feat_masks


class TransformerV1Sampler(BaseSampler):
    def __init__(
        self,
        efficiency,
        start_idx,
        end_idx,
        dim,
        niters,
        loss_type,
        efficiency_offset,
        disable_stepwise_loss,
        mask_prev,
        use_video_mask_for_loss,
    ):
        super().__init__(efficiency, start_idx, end_idx)
        self.efficiency_offset = efficiency_offset
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
        src_vid_total,
        src_txt=None,
        src_vid_mask_total=None,
        src_txt_mask=None,
        pos_txt=None,
        input_vid_proj=None,
        position_embed=None,
        cross_self_encoder=None,
        gs_tau=0.01,
        niters=None,
        cache=None,
    ):
        # src_vid_total - list of (B, L/N, F) tensors
        split_len = src_vid_total[0].shape[1]
        src_vid_total = torch.cat(src_vid_total, dim=1)
        src_vid_mask_total = torch.cat(src_vid_mask_total, dim=1)
        if cache:
            sampler_masks = cache["sampler_masks"]  # (B, niters, L)
            Fh = self.end_idx - self.start_idx + 1
            sampler_masks_sum = torch.clamp(sampler_masks.sum(dim=1), 0.0, 1.0)
            final_mask = torch.ones_like(src_vid_total)
            final_mask[:, :, :Fh] = sampler_masks_sum.unsqueeze(2)
            src_vid_total_samp = src_vid_total * final_mask
        else:
            src_vid_total_samp, sampler_masks = self.sample_features(
                src_vid_total,
                src_txt,
                src_vid_mask_total,
                src_txt_mask,
                pos_txt,
                input_vid_proj,
                position_embed,
                cross_self_encoder,
                gs_tau,
                niters=niters,
            )
        src_vid_total_samp = torch.split(src_vid_total_samp, split_len, dim=1)
        return src_vid_total_samp, sampler_masks

    def sample_features(
        self,
        src_vid,
        src_txt,
        src_vid_mask,
        src_txt_mask,
        pos_txt,
        input_vid_proj,
        position_embed,
        cross_self_encoder,
        gs_tau,
        niters=None,
    ):
        # src_vid - (B, L, F)
        # src_vid_mask - (B, L)
        B, L, F = src_vid.shape

        fine_features = src_vid[:, :, self.start_idx : self.end_idx + 1]
        coarse_features = src_vid[:, :, self.end_idx + 1 :]
        # Compute scores iteratively
        predicted_masks = []
        niters = self.niters if niters is None else niters
        all_mask = torch.zeros_like(src_vid_mask)
        for i in range(niters):
            src_vid_i = torch.cat(
                [fine_features * all_mask.unsqueeze(2), coarse_features], dim=2
            )
            src_vid_i = input_vid_proj(src_vid_i)
            pos_vid_i = position_embed(src_vid_i, src_vid_mask)
            vid_mem_i, txt_mem_i = cross_self_encoder(
                src_vid_i, src_vid_mask, src_txt, src_txt_mask, pos_vid_i, pos_txt
            )
            vid_mem_i = vid_mem_i + (txt_mem_i * 0.0).sum()  # (B, L, F)
            selection_logits = self.selection_mlp(vid_mem_i)  # (B, L, 2)
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
            all_mask = all_mask + per_step_mask * src_vid_mask
            if not self.mask_prev:
                all_mask = torch.clamp(all_mask, 0.0, 1.0)
            predicted_masks.append(per_step_mask * src_vid_mask)

        src_vid_sampled = torch.cat(
            [fine_features * all_mask.unsqueeze(2), coarse_features], dim=2
        )
        predicted_masks = torch.stack(predicted_masks, dim=1)  # (B, niters, L)
        return src_vid_sampled, predicted_masks

    def compute_loss(self, predicted_masks, video_mask):
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
        # video_mask - (B, L)
        _, niters, L = predicted_masks.shape
        assert niters <= self.niters
        nsamples_total = (1 - self.efficiency - self.efficiency_offset) * L
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
        total_masks = torch.clamp(predicted_masks.sum(dim=1), 0.0, 1.0).unsqueeze(2)
        sampling_rate = super().compute_sampling_rate(total_masks, video_masks)
        return sampling_rate

    @classmethod
    def from_configs(cls, configs):
        return cls(
            configs.feature_sampler_efficiency,
            configs.feature_mask_idxs[0],
            configs.feature_mask_idxs[1],
            configs.cross_hidden_dim,
            configs.sampler_niters,
            configs.sampler_loss_type,
            configs.feature_sampler_efficiency_offset,
            configs.disable_stepwise_loss,
            configs.sampler_mask_prev,
            configs.video_mask_for_sampler_loss,
        )


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
        src_vid_total,
        src_txt=None,
        src_vid_mask_total=None,
        src_txt_mask=None,
        gs_tau=0.01,
        cache=None,
        *args,
        **kwargs,
    ):
        # src_vid_total - list of (B, L/N, F) tensors
        split_len = src_vid_total[0].shape[1]
        src_vid_total = torch.cat(src_vid_total, dim=1)
        src_vid_mask_total = torch.cat(src_vid_mask_total, dim=1)
        if cache:
            sampler_masks = cache["sampler_masks"]  # (B, L)
            Fh = self.end_idx - self.start_idx + 1
            final_mask = torch.ones_like(src_vid_total)
            final_mask[:, :, :Fh] = sampler_masks.unsqueeze(2)
            src_vid_total_samp = src_vid_total * final_mask
        else:
            src_vid_total_samp, sampler_masks = self.sample_features(
                src_vid_total,
                src_txt,
                src_vid_mask_total,
                src_txt_mask,
                gs_tau,
            )
        src_vid_total_samp = torch.split(src_vid_total_samp, split_len, dim=1)
        return src_vid_total_samp, sampler_masks

    def sample_features(
        self,
        video_features,
        query_features,
        v_mask,
        q_mask,
        gs_tau,
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

    def compute_loss(self, predicted_masks, video_mask):
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
        # video_masks - (B, L)
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
            configs.v_feat_dim,
            configs.cross_hidden_dim,
            configs.sampler_coarse_hdim,
            configs.sampler_fine_hdim,
            configs.sampler_loss_type,
            configs.video_mask_for_sampler_loss,
        )

    def compute_sampling_rate(self, predicted_masks, video_masks):
        """
        Args:
            predicted_masks - (B, L)
            video_masks - (B, L)
        """
        return super().compute_sampling_rate(predicted_masks.unsqueeze(2), video_masks)


class OCSampler(BaseSampler):
    def __init__(
        self,
        efficiency,
        start_idx,
        end_idx,
        v_dim,
        q_dim,
    ):
        super().__init__(efficiency, start_idx, end_idx)
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
        src_vid_total,
        src_txt=None,
        src_vid_mask_total=None,
        src_txt_mask=None,
        gs_tau=0.01,
        cache=None,
        *args,
        **kwargs,
    ):
        # src_vid_total - list of (B, L/N, F) tensors
        split_len = src_vid_total[0].shape[1]
        src_vid_total = torch.cat(src_vid_total, dim=1)
        src_vid_mask_total = torch.cat(src_vid_mask_total, dim=1)
        if cache:
            sampler_masks = cache["sampler_masks"]  # (B, L)
            Fh = self.end_idx - self.start_idx + 1
            final_mask = torch.ones_like(src_vid_total)
            final_mask[:, :, :Fh] = sampler_masks.unsqueeze(2)
            src_vid_total_samp = src_vid_total * final_mask
        else:
            src_vid_total_samp, sampler_masks = self.sample_features(
                src_vid_total,
                src_txt,
                src_vid_mask_total,
                src_txt_mask,
                gs_tau,
            )
        src_vid_total_samp = torch.split(src_vid_total_samp, split_len, dim=1)
        return src_vid_total_samp, sampler_masks

    def sample_features(
        self,
        video_features,
        query_features,
        v_mask,
        q_mask,
        gs_tau,
    ):
        # video_features - (B, Lv, Fv)
        # query_features - (B, Lq, Fq)
        # v_mask - (B, Lv)
        # q_mask - (B, Lq)
        B, L, F = video_features.shape

        k = int(math.floor((1 - self.efficiency) * L))

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
        topk_mask = self.topk_operator(scores, k, gs_tau)  # (B, L)
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
            configs.v_feat_dim,
            configs.cross_hidden_dim,
        )

    def compute_sampling_rate(self, predicted_masks, video_masks):
        """
        Args:
            predicted_masks - (B, L)
            video_masks - (B, L)
        """
        return super().compute_sampling_rate(predicted_masks.unsqueeze(2), video_masks)
