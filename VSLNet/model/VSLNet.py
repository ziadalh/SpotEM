"""VSLNet Baseline for Ego4D Episodic Memory -- Natural Language Queries."""

import torch
import torch.nn as nn
from model.layers import (
    BertEmbedding,
    ConditionedPredictor,
    CQAttention,
    CQConcatenate,
    DistilBertEmbedding,
    Embedding,
    FeatureEncoder,
    HighLightLayer,
    SplitVisualProjection,
    VisualProjection,
)
from model.samplers import (
    AllSampler,
    LiteEvalSampler,
    OCSampler,
    RandomSampler,
    TransformerV1Sampler,
    UniformSampler,
    ZeroSampler,
)
from transformers import AdamW, get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(model, configs):
    no_decay = [
        "bias",
        "layer_norm",
        "LayerNorm",
    ]  # no decay for parameters of layer norm and bias
    if configs.use_feature_sampler:
        sampler_params_wd = []
        sampler_params_nowd = []
        task_params_wd = []
        task_params_nowd = []
        for n, p in model.named_parameters():
            if not any(nd in n for nd in no_decay):
                if n.startswith("sampler."):
                    sampler_params_wd.append(p)
                else:
                    task_params_wd.append(p)
            else:
                if n.startswith("sampler."):
                    sampler_params_nowd.append(p)
                else:
                    task_params_nowd.append(p)
        optimizer_grouped_parameters = [
            {
                "params": sampler_params_wd,
                "weight_decay": 0.01,
                "lr": configs.init_lr * configs.lr_scale_sampler,
            },
            {
                "params": task_params_wd,
                "weight_decay": 0.01,
            },
            {
                "params": sampler_params_nowd,
                "weight_decay": 0.0,
                "lr": configs.init_lr * configs.lr_scale_sampler,
            },
            {
                "params": task_params_nowd,
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        configs.num_train_steps * configs.warmup_proportion,
        configs.num_train_steps,
    )
    return optimizer, scheduler


class VSLNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__()
        self.configs = configs
        if self.configs.use_split_visual_projection:
            assert len(self.configs.feature_split_points) > 0
            self.video_affine = SplitVisualProjection(
                visual_dim=configs.video_feature_dim,
                split_points=configs.feature_split_points,
                split_dims=configs.feature_split_dims,
                dim=configs.dim,
                drop_rate=configs.drop_rate,
            )
        else:
            self.video_affine = VisualProjection(
                visual_dim=configs.video_feature_dim,
                dim=configs.dim,
                drop_rate=configs.drop_rate,
            )
        self.feature_encoder = FeatureEncoder(
            dim=configs.dim,
            num_heads=configs.num_heads,
            kernel_size=7,
            num_layers=4,
            max_pos_len=configs.max_pos_len,
            drop_rate=configs.drop_rate,
        )
        # video and query fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat = CQConcatenate(dim=configs.dim)
        # query-guided highlighting
        self.highlight_layer = HighLightLayer(dim=configs.dim)
        # conditioned predictor
        self.predictor = ConditionedPredictor(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len,
            predictor=configs.predictor,
        )

        # If use samplers, create the sampler module
        self.sampler = None
        if configs.use_feature_sampler:
            if configs.feature_sampler_type == "random":
                sampler_cls = RandomSampler
            elif configs.feature_sampler_type == "uniform":
                sampler_cls = UniformSampler
            elif configs.feature_sampler_type == "zero":
                sampler_cls = ZeroSampler
                assert configs.feature_sampler_efficiency == 1
            elif configs.feature_sampler_type == "all":
                sampler_cls = AllSampler
                assert configs.feature_sampler_efficiency == 0
            elif configs.feature_sampler_type == "transformer-v1":
                sampler_cls = TransformerV1Sampler
            elif configs.feature_sampler_type == "liteeval":
                sampler_cls = LiteEvalSampler
            elif configs.feature_sampler_type == "ocsampler":
                sampler_cls = OCSampler

            self.sampler = sampler_cls.from_configs(configs)

        # If pretrained transformer, initialize_parameters and load.
        if configs.predictor == "bert":
            # Project back from BERT to dim.
            assert configs.query_feature_dim == 768
            self.query_affine = nn.Linear(768, configs.dim)
            # init parameters
            self.init_parameters()
            self.embedding_net = BertEmbedding(configs.text_agnostic)
        elif configs.predictor == "egovlp-distilbert":
            # Project back from BERT to dim.
            assert configs.query_feature_dim == 768
            self.query_affine = nn.Linear(768, configs.dim)
            # init parameters
            self.init_parameters()
            self.embedding_net = DistilBertEmbedding(configs.text_agnostic)
            # load egovlp weights
            state_dict = torch.load(configs.egovlp_predictor_ckpt)
            self.embedding_net.embedder.load_state_dict(state_dict)
        else:
            self.embedding_net = Embedding(
                num_words=configs.word_size,
                num_chars=configs.char_size,
                out_dim=configs.dim,
                word_dim=configs.word_dim,
                char_dim=configs.char_dim,
                word_vectors=word_vectors,
                drop_rate=configs.drop_rate,
            )
            # init parameters
            self.init_parameters()
        if configs.resume != "":
            state_dict = torch.load(configs.resume)
            self.load_state_dict(state_dict, strict=False)

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
        word_ids,
        char_ids,
        video_features,
        v_mask,
        q_mask,
        gs_tau=0.01,
        niters=None,
        return_features=False,
    ):
        if self.configs.predictor in ["bert", "egovlp-distilbert"]:
            query_features = self.embedding_net(word_ids)
            query_features = self.query_affine(query_features)
        else:
            query_features = self.embedding_net(word_ids, char_ids)
        if query_features.shape[1] >= self.configs.max_pos_len:
            query_features = query_features[:, : self.configs.max_pos_len]
            q_mask = q_mask[:, : self.configs.max_pos_len]

        sampler_masks = None
        if self.sampler:
            video_features, sampler_masks = self.sampler(
                video_features,
                query_features=query_features,
                v_mask=v_mask,
                q_mask=q_mask,
                video_affine=self.video_affine,
                feature_encoder=self.feature_encoder,
                cq_attention=self.cq_attention,
                cq_concat=self.cq_concat,
                gs_tau=gs_tau,
                niters=niters,
            )
        video_features = self.video_affine(video_features)
        query_features = self.feature_encoder(query_features, mask=q_mask)
        video_features = self.feature_encoder(video_features, mask=v_mask)
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        h_score = self.highlight_layer(features, v_mask)
        h_features = features * h_score.unsqueeze(2)
        start_logits, end_logits = self.predictor(h_features, mask=v_mask)
        if return_features:
            return h_score, start_logits, end_logits, features, sampler_masks
        else:
            return h_score, start_logits, end_logits, sampler_masks

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(
            start_logits=start_logits, end_logits=end_logits
        )

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(
            scores=scores, labels=labels, mask=mask
        )

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
        )

    def compute_sampler_loss(self, sampler_masks, video_mask):
        if self.sampler:
            return self.sampler.compute_loss(sampler_masks, video_mask)
        else:
            return 0.0

    def compute_sampling_rate(self, sampler_masks, video_mask):
        if self.sampler:
            return self.sampler.compute_sampling_rate(sampler_masks, video_mask)
        else:
            return torch.ones(video_mask.shape[0], device=video_mask.device)

    def compute_highlight_distillation_loss(self, scores_l, scores_e, mask):
        return self.highlight_layer.compute_distillation_loss(scores_l, scores_e, mask)

    def compute_predictor_distillation_loss(
        self, start_logits_l, end_logits_l, start_logits_e, end_logits_e
    ):
        return self.predictor.compute_distillation_loss(
            start_logits_l, end_logits_l, start_logits_e, end_logits_e
        )

    def compute_feature_distillation_loss(self, features_l, features_e, mask):
        loss = nn.L1Loss(reduction="none")(features_l, features_e).mean(dim=2)  # (B, L)
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-6)
        return loss
