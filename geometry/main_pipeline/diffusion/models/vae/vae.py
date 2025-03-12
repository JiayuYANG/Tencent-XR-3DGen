import json
import sys
import torch
import os
from einops import repeat
from easydict import EasyDict as edict
# try:
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, "CraftsMan3D"))
from craftsman.models.autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder
from craftsman.utils.config import ExperimentConfig, load_config
# except:
#     exit("Error: please clone CraftsMan3D first from https://github.com/wyysf-98/CraftsMan3D.git")

class MichelangeloAutoencoderSparse(MichelangeloAutoencoder):
    def sparse_query(self, xyz_samples, latents, num_chunks=10000):
        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start: start + num_chunks, :].to(latents)
            batch_queries = repeat(queries, "p c -> b p c", b=1)

            logits = self.query(batch_queries, latents)
            batch_logits.append(logits.cpu())
        
        res_logits = torch.cat(batch_logits, 1).squeeze(0)
        return res_logits


def get_vae_model(specs):
    vae_config = specs["vae_config"]
    if vae_config["vae_type"] == "craftsman_vae":
        crm_config_path = vae_config["config_path"]
        cfg = load_config(crm_config_path)
        vae_model = MichelangeloAutoencoderSparse(cfg.system.shape_model).eval()

    return vae_model