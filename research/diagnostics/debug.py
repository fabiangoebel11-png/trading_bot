import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from core.config import TradingBotConfig
from core.ml.tcn_model import TCNTrendModel

config = TradingBotConfig().ml
config.batch_size = 3
config.use_amp = False
config.hidden_channels = 4
config.num_layers = 1
config.dropout = 0.0
wrapper = TCNTrendModel(config, n_features=2)

class DummyModel(torch.nn.Module):
    def forward_outputs(self, inputs):
        first = inputs[:, 0, 0]
        logits = torch.stack((first, -first, torch.zeros_like(first)), dim=1)
        return {"logits": logits, "expected_mfe": first, "expected_mae": -first}

wrapper.model = DummyModel().to(wrapper.device)
X = np.arange(10 * 2, dtype=np.float32).reshape(10, 1, 2) / 10.0
y = np.arange(10, dtype=np.int64) % 3
mfe = X[:, 0, 0].astype(np.float32)
mae = -mfe
print('batched', wrapper._evaluate_validation_loss(X, y, mfe, mae))
with torch.no_grad():
    outputs = wrapper.model.forward_outputs(torch.tensor(X, device=wrapper.device))
    ce = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, reduction='none')(outputs['logits'], torch.tensor(y, device=wrapper.device))
    reg = torch.nn.SmoothL1Loss(reduction='none')
    direction_term = config.direction_loss_weight * ce.mean()
    mfe_term = config.mfe_loss_weight * reg(outputs['expected_mfe'], torch.tensor(mfe, device=wrapper.device)).mean()
    mae_term = config.mae_loss_weight * reg(outputs['expected_mae'], torch.tensor(mae, device=wrapper.device)).mean()
    weights = config.direction_loss_weight + config.mfe_loss_weight + config.mae_loss_weight
    direct = (direction_term + mfe_term + mae_term) / weights
    print('ce mean', ce.mean())
    print('mfe mean', reg(outputs['expected_mfe'], torch.tensor(mfe, device=wrapper.device)).mean())
    print('mae mean', reg(outputs['expected_mae'], torch.tensor(mae, device=wrapper.device)).mean())
    print('direct', direct)
    for start in range(0, len(X), config.batch_size):
        stop = start + config.batch_size
        batch_x = torch.from_numpy(wrapper._normalize_batch(X[start:stop])).to(wrapper.device)
        batch_y = torch.from_numpy(y[start:stop].astype(np.int64, copy=False)).to(wrapper.device)
        batch_mfe = torch.from_numpy(np.asarray(mfe[start:stop], dtype=np.float32, order='C', copy=False) / wrapper.mfe_target_scale).to(wrapper.device)
        batch_mae = torch.from_numpy(np.asarray(mae[start:stop], dtype=np.float32, order='C', copy=False) / wrapper.mae_target_scale).to(wrapper.device)
        outputs = wrapper.model.forward_outputs(batch_x)
        direction_loss = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, reduction='none')(outputs['logits'], batch_y).mean()
        mfe_loss = torch.nn.SmoothL1Loss(reduction='none')(outputs['expected_mfe'], batch_mfe).mean()
        mae_loss = torch.nn.SmoothL1Loss(reduction='none')(outputs['expected_mae'], batch_mae).mean()
        combo = (config.direction_loss_weight * direction_loss + config.mfe_loss_weight * mfe_loss + config.mae_loss_weight * mae_loss) / (config.direction_loss_weight + config.mfe_loss_weight + config.mae_loss_weight)
        print(start, 'dir', direction_loss.item(), 'mfe', mfe_loss.item(), 'mae', mae_loss.item(), 'combo', combo.item())
