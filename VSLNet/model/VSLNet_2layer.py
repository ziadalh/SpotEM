import copy

import torch

from .VSLNet import VSLNet


class VSLNetTwoLayer(VSLNet):
    def __init__(self, configs, word_vectors, device):
        super().__init__(configs, word_vectors)
        # Sanity check
        assert not configs.use_feature_sampler

        layer1_configs = copy.deepcopy(configs)
        layer1_configs.use_feature_sampler = True
        layer1_configs.feature_sampler_type = "zero"
        layer1_configs.feature_sampler_efficiency = 1.0

        self.layer1_model = [VSLNet(layer1_configs, word_vectors)]
        self.layer1_model[0].eval()
        self.layer1_model[0].to(device)
        ckpt = torch.load(configs.pretrained_zeroclip_path)
        self.layer1_model[0].load_state_dict(ckpt)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask, **kwargs):
        _, start_logits, end_logits, _ = self.layer1_model[0](
            word_ids, char_ids, video_features, v_mask, q_mask
        )
        start_indices, end_indices = self.layer1_model[0].extract_index(
            start_logits, end_logits
        )
        start_indices = start_indices.cpu().numpy()
        end_indices = end_indices.cpu().numpy()

        sample_masks = torch.zeros_like(v_mask)  # (B, L)
        topk = self.configs.twolayer_topk
        assert topk <= len(start_indices)

        for b, (starts, ends) in enumerate(zip(start_indices, end_indices)):
            for i, (start, end) in enumerate(zip(starts, ends)):
                if i >= topk:
                    continue
                sample_masks[b, start : end + 1] = 1

        # Mask out unselected features
        fstart, fend = (
            self.configs.feature_mask_idxs[0],
            self.configs.feature_mask_idxs[1],
        )
        fine_features = video_features[:, :, fstart : fend + 1]
        coarse_features = video_features[:, :, fend + 1 :]
        sampled_video_features = torch.cat(
            [
                fine_features * sample_masks.unsqueeze(2),
                coarse_features,
            ],
            dim=2,
        )  # (B, L, F)
        outputs = super().forward(
            word_ids, char_ids, sampled_video_features, v_mask, q_mask, **kwargs
        )
        outputs = tuple([*outputs[:-1], sample_masks])
        return outputs

    def compute_sampling_rate(self, sampler_masks, video_mask):
        return self.layer1_model[0].sampler.compute_sampling_rate(
            sampler_masks, video_mask
        )
