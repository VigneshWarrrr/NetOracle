"""
Explainability Module: Feature Attribution and Attention Profiling
"""
import torch
import numpy as np
from typing import List, Tuple

class WorldModelExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
        self.full_feature_names = []
        for stat in ['mean', 'std', 'max', 'min']:
            for name in feature_names:
                self.full_feature_names.append(f"{name}_{stat}")
        self.full_feature_names.append("flow_count")

    def get_top_features(self, x_tensor: torch.Tensor, top_k: int = 10) -> List[Tuple[str, float]]:
        self.model.eval()
        x_clone = x_tensor.clone().detach().requires_grad_(True)
        out = self.model(x_clone)
        score = out['attack_probs'].sum()
        score.backward()
        
        attr = (x_clone.grad * x_clone).abs().mean(dim=(0, 1)).detach().cpu().numpy()
        ranked = []
        for i in range(min(len(attr), len(self.full_feature_names))):
            ranked.append((self.full_feature_names[i], float(attr[i])))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
