import torch
from torch import nn
import torch.nn.functional as F 
from torch.nn import LayerNorm
import numpy as np
from tfm_model import TemporalEncoder, TemporalDecoder, get_position_embedding_sine, QuickGELU
from collections import OrderedDict

class ExoGroundingTransformer(nn.Module):
    def __init__(self, 
                 num_encoder_layers=2, 
                 num_decoder_layers=2,
                 video_embed_dim=4096,
                 text_embed_dim=4096,
                 pose_embed_dim=72,
                 feature_dim=512,
                 num_max_views=1,
                 mode='pose+video', # options: 'pose_only', 'pose+video", 'video_only'
                 ):
        super().__init__()

        #initialize args
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.text_embed_dim = text_embed_dim
        self.video_embed_dim = video_embed_dim
        self.pose_embed_dim = pose_embed_dim
        self.feature_dim = feature_dim
        self.num_max_views = num_max_views
        self.mode = mode

        #initalize multi-modal encoder and narration decoder
        self.tfm_modules = []
        self.multi_modal_encoder = TemporalEncoder(width=feature_dim, layers=self.num_encoder_layers, heads=8) #transformer
        self.tfm_modules.append(self.multi_modal_encoder)

        self.decoder = TemporalDecoder(width=feature_dim, layers=num_decoder_layers, heads=8)
        self.tfm_modules.append(self.decoder)
        self.grounding_head = nn.Linear(self.feature_dim, 2)
        self.activation = nn.Sigmoid()
        
        self.video_unimodal_encoder = TemporalEncoder(width=feature_dim, layers=self.num_encoder_layers, heads=8)
        self.tfm_modules.append(self.video_unimodal_encoder)
        self.text_unimodal_encoder = TemporalEncoder(width=feature_dim, layers=self.num_encoder_layers, heads=8)
        self.tfm_modules.append(self.text_unimodal_encoder)

        self.pose_pre_proj = nn.Linear(self.pose_embed_dim, self.feature_dim, bias=False)
        self.ln_pose_init = LayerNorm(self.feature_dim)
        self.temporal_pos_embed = nn.Parameter(torch.empty(1024, self.feature_dim))
        self.ln_position_init = LayerNorm(self.feature_dim)
        self.video_pre_proj = nn.Linear(self.video_embed_dim, self.feature_dim, bias=False)
        self.ln_video_init = LayerNorm(self.feature_dim)

        # temporal positional encoding for video
        nn.init.normal_(self.temporal_pos_embed, std=0.01)

        #initialize embeddings and projection layers
        self.text_pre_proj = nn.Linear(self.text_embed_dim, self.feature_dim, bias=False)
        self.ln_text_init = LayerNorm(self.feature_dim)
        self.ln_joint_post_enc = LayerNorm(self.feature_dim)
        self.ln_video_post_enc = LayerNorm(self.feature_dim)
        self.ln_text_post_enc = LayerNorm(self.feature_dim)
        

        # temporal positional encoding for text
        self.text_temporal_pos_embed = nn.Parameter(torch.empty(self.text_embed_dim, self.feature_dim))
        nn.init.normal_(self.text_temporal_pos_embed, std=0.01)

        self.initialize_parameters()

    def initialize_parameters(self):
        if self.mode == 'pose+video':
            linear_layers = [self.video_pre_proj, self.text_pre_proj, self.pose_pre_proj, self.grounding_head]
        elif self.mode == 'video_only':
            linear_layers = [self.video_pre_proj, self.text_pre_proj, self.grounding_head]
        elif self.mode == 'pose_only':
            linear_layers = [self.text_pre_proj, self.pose_pre_proj, self.grounding_head]

        for layer in linear_layers:
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for tfm_module in self.tfm_modules:
            proj_std = (tfm_module.width ** -0.5) * ((2 * tfm_module.layers) ** -0.5)
            attn_std = tfm_module.width ** -0.5
            fc_std = (2 * tfm_module.width) ** -0.5
            for block in tfm_module.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, video_embed, lang_embed,
                video_padding_mask, lang_padding_mask, pose_embed=None):
        # text embedding without temporal-enc
        lang_embed_raw = self.get_textual_feature(lang_embed) #nn.Linear + Layernorm
        lang_embed_with_time = lang_embed_raw
        N = lang_embed_with_time.shape[1] #number of narration #equivalent to N=1

        # encode video features and text features separately
        video_encoded_features = self.get_unimodal_features("video", video_embed, video_padding_mask)
        video_encoded_features = video_encoded_features.mean(dim=1) #Linear + Layernorm + position embedding + Transformers #[B,30,512]
        text_encoded_features = self.get_unimodal_features("text", lang_embed_with_time, lang_padding_mask).mean(dim=1) #Transformer

        #print('video_padding_mask', video_padding_mask)

        # combine with pose
        if self.mode in ('pose_only', 'pose+video'):
            pose_encoded_features = self.get_pose_feature(pose_embed)

        if self.mode == 'pose+video':
            video_encoded_features = torch.cat((video_encoded_features, pose_encoded_features), dim=1) # [B,60,512]
            video_padding_mask = torch.cat((video_padding_mask, video_padding_mask), dim=1)

        # get multi-modal feature output from encoder   
        if self.mode in ('video_only', 'pose+video'):
            all_output, _ = self.get_joint_feature(
                video_encoded_features.squeeze(dim=1), video_padding_mask,
                text_encoded_features, lang_padding_mask)  #concat + Transformer
        elif self.mode == 'pose_only':
            all_output, _ = self.get_joint_feature(
                pose_encoded_features.squeeze(dim=1), video_padding_mask,
                text_encoded_features, lang_padding_mask)  #concat + Transformer

        decoder_context = all_output[:, :, :-N]
        text_features = all_output[:, :, -N:]

        # decoder for prediction
        decoder_output = self.decoder(x=text_features[:,-1,::].permute(1, 0, 2), memory=decoder_context[:,-1,::].permute(1, 0, 2), tgt_key_padding_mask=lang_padding_mask, memory_key_padding_mask=video_padding_mask)
        decoder_features = decoder_output[-1].permute(1,0,2)
        #grounding = self.activation(self.grounding_head(decoder_features))
        grounding = self.grounding_head(decoder_features)

        output_dict = {'interval_preds': grounding, 'low_dim_features': video_encoded_features}
        return output_dict 

    def get_unimodal_features(self, modality, feat_embed, padding_mask):
        B,T,_,= feat_embed.shape

        if modality == "video":
            proj_embed = self.ln_video_init(self.video_pre_proj(feat_embed))
            seq_len = T // self.num_max_views
            pos_start_idx = np.random.randint(0, int(seq_len/2))
            pos_embed = self.temporal_pos_embed[None, pos_start_idx:pos_start_idx+seq_len, :]
            pos_embed = pos_embed.repeat(1, self.num_max_views, 1)
            feat_embed_with_time = proj_embed + self.ln_position_init(pos_embed)
        else:
            feat_embed_with_time = feat_embed

        feat_embed_with_time = feat_embed_with_time.permute(1,0,2) # BXC -> XBC

        if modality == "video":    
            feat_output = self.video_unimodal_encoder(feat_embed_with_time, padding_mask)
            feat_output[-1] = self.ln_video_post_enc(feat_output[-1])
        else:
            feat_output = self.text_unimodal_encoder(feat_embed_with_time, padding_mask)
            feat_output[-1] =self.ln_text_post_enc(feat_output[-1])
        feat_output = torch.stack(feat_output, dim=1).permute(2,1,0,3)  # B,Stage,X,C

        return feat_output

    def get_joint_feature(self, video_embed_with_time, video_padding_mask,
                          lang_embed_with_time, lang_padding_mask):
        """Get the joint video embedding and text embedding from the joint encoder.
        It takes both visual and textual inputs."""
        B,T,_,= video_embed_with_time.shape
        seq_len = T // self.num_max_views

        pos_start_idx = np.random.randint(0, int(seq_len/2))
        pos_embed = self.temporal_pos_embed[None, pos_start_idx:pos_start_idx+seq_len, :]
        pos_embed = pos_embed.repeat(1, self.num_max_views, 1)

        joint_embed = torch.cat((video_embed_with_time, lang_embed_with_time), dim=1)
        joint_embed = joint_embed.permute(1,0,2) # BXC -> XBC

        joint_padding_mask = torch.cat((video_padding_mask, lang_padding_mask), dim=1)

        joint_output = self.multi_modal_encoder(joint_embed, joint_padding_mask)
        joint_output[-1] = self.ln_joint_post_enc(joint_output[-1])
        joint_output = torch.stack(joint_output, dim=1).permute(2,1,0,3)  # B,Stage,X,C

        return joint_output, T

    def get_textual_feature(self, lang_embed):
        """get text embedding after proj and LayerNorm"""
        text_proj = self.ln_text_init(self.text_pre_proj(lang_embed))
        return text_proj

    def get_pose_feature(self, pose_embed):
        """get pose embedding after proj and LayerNorm"""
        pose_proj = self.ln_pose_init(self.pose_pre_proj(pose_embed))
        return pose_proj


class SimilarityModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, video_embed, lang_embed, delta):
        
        B, T, D = video_embed.shape
        lang_embed = lang_embed.expand(-1, T, -1)  # [B, T, D]

        # Normalize embeddings to compute cosine similarity
        video_norm = F.normalize(video_embed, dim=-1)  # [B, T, D]
        lang_norm = F.normalize(lang_embed, dim=-1)    # [B, T, D]

        sim = (video_norm * lang_norm).sum(dim=-1)  # [B, T] cosine similarity per frame

        regions = []
        for i in range(B):
            sim_i = sim[i]  # [T]
            peak_idx = torch.argmax(sim_i).item()
            peak_val = sim_i[peak_idx].item()

            # Expand left
            start = peak_idx
            while start > 0 and abs(sim_i[start - 1].item() - peak_val) <= delta:
                start -= 1
            if start != peak_idx:
                start += 1

            # Expand right
            end = peak_idx
            while end < T - 1 and abs(sim_i[end + 1].item() - peak_val) <= delta:
                end += 1
            if end != peak_idx:
                end -= 1

            regions.append((start, end))

        return torch.tensor([a[0] for a in regions]), torch.tensor([a[1] for a in regions])
