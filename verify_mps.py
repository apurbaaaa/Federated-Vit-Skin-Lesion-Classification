#!/usr/bin/env python3
"""
End-to-end verification on MPS with synthetic data.
Tests: model forward, loss, EMA, MixUp/CutMix, scheduler, TTA, checkpoint.
"""

import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

# -- Config --
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("mps")
BS = 2
IMG_SIZE = 384
NUM_CLASSES = 8
META_DIM = 13

results = {}

def test(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results[name] = "PASS"
                print(f"  ✅ {name}")
            except Exception as e:
                results[name] = f"FAIL: {e}"
                print(f"  ❌ {name}: {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ============================================================
# 1. Model forward pass
# ============================================================
@test("model_forward")
def test_model_forward():
    from model import build_model
    model = build_model(config).to(device)
    x = torch.randn(BS, 4, IMG_SIZE, IMG_SIZE, device=device)
    meta = torch.randn(BS, META_DIM, device=device)
    out = model(x, metadata=meta)
    assert "logits" in out, "Missing 'logits' key"
    assert out["logits"].shape == (BS, NUM_CLASSES), f"Wrong shape: {out['logits'].shape}"
    # Without metadata
    out2 = model(x, metadata=None)
    assert out2["logits"].shape == (BS, NUM_CLASSES)

test_model_forward()


# ============================================================
# 2. Loss
# ============================================================
@test("asymmetric_focal_loss")
def test_loss():
    from losses import build_loss
    criterion = build_loss(config).to(device)
    logits = torch.randn(BS, NUM_CLASSES, device=device)
    labels = torch.randint(0, NUM_CLASSES, (BS,), device=device)
    loss = criterion(logits, labels)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"

test_loss()


# ============================================================
# 3. EMA
# ============================================================
@test("ema")
def test_ema():
    from model import build_model
    from utils import EMA
    model = build_model(config).to(device)
    ema = EMA(model, decay=0.9995)
    # Simulate a step
    x = torch.randn(1, 4, IMG_SIZE, IMG_SIZE, device=device)
    out = model(x)
    out["logits"].sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            p.data -= 0.01 * p.grad
            break
    ema.update()
    # apply/restore
    ema.apply_shadow()
    ema.restore()
    # state dict roundtrip
    sd = ema.state_dict()
    ema.load_state_dict(sd)

test_ema()


# ============================================================
# 4. MixUp / CutMix
# ============================================================
@test("mixup_cutmix")
def test_mixup():
    from utils import MixUp, CutMix, MixupCutmix, mixup_criterion
    from losses import build_loss
    images = torch.randn(BS, 4, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (BS,))
    
    mu = MixUp(alpha=0.4)
    mixed, la, lb, lam = mu(images, labels)
    assert mixed.shape == images.shape

    cm = CutMix(alpha=1.0, prob=1.0)
    mixed, la, lb, lam = cm(images, labels)
    assert mixed.shape == images.shape

    mc = MixupCutmix(mixup_alpha=0.4, cutmix_alpha=1.0, cutmix_prob=0.7)
    mixed, la, lb, lam = mc(images, labels)
    
    criterion = build_loss(config)
    logits = torch.randn(BS, NUM_CLASSES)
    loss = mixup_criterion(criterion, logits, la, lb, lam)
    assert loss.dim() == 0

test_mixup()


# ============================================================
# 5. Scheduler
# ============================================================
@test("warmup_cosine_scheduler")
def test_scheduler():
    from utils import WarmupCosineScheduler
    model_params = [torch.nn.Parameter(torch.randn(10))]
    opt = torch.optim.AdamW(model_params, lr=1e-4)
    sched = WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=80, min_lr=1e-6)
    lrs = []
    for _ in range(80):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] < lrs[4], "LR should increase during warmup"
    assert lrs[-1] <= lrs[5], "LR should decrease after warmup"

test_scheduler()


# ============================================================
# 6. Train transform + dataset
# ============================================================
@test("train_transform")
def test_train_tf():
    from data import TrainTransform
    from PIL import Image
    tf = TrainTransform(384, cfg=config.get("augmentation", {}).get("train", {}))
    img = Image.fromarray(np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8))
    mask = Image.fromarray(np.random.randint(0, 255, (450, 600), dtype=np.uint8))
    img_t, mask_t = tf(img, mask)
    assert img_t.shape == (3, 384, 384), f"Image shape: {img_t.shape}"
    assert mask_t.shape == (1, 384, 384), f"Mask shape: {mask_t.shape}"

test_train_tf()


@test("eval_transform")
def test_eval_tf():
    from data import EvalTransform
    from PIL import Image
    tf = EvalTransform(384)
    img = Image.fromarray(np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8))
    img_t, _ = tf(img, None)
    assert img_t.shape == (3, 384, 384)

test_eval_tf()


# ============================================================
# 7. Metadata encoding
# ============================================================
@test("metadata_encoding")
def test_meta():
    from data import encode_metadata_vector, META_DIM
    vec = encode_metadata_vector(0.5, 1, 3)
    assert vec.shape == (META_DIM,), f"Meta shape: {vec.shape}"
    assert vec[0] == 0.5  # age
    assert vec[2] == 1.0  # sex=female
    assert vec[1 + 3 + 3] == 1.0  # site=posterior torso (idx 3)

test_meta()


# ============================================================
# 8. ISICDataset with synthetic data
# ============================================================
@test("isic_dataset_synthetic")
def test_dataset():
    from data import ISICDataset
    # Create synthetic dataframe
    tmpdir = tempfile.mkdtemp()
    from PIL import Image
    rows = []
    for i in range(4):
        img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
        path = os.path.join(tmpdir, f"img_{i}.jpg")
        img.save(path)
        rows.append({
            "image": f"img_{i}",
            "image_path": path,
            "label": i % NUM_CLASSES,
            "age_norm": 0.5,
            "sex_idx": 1,
            "site_idx": 2,
        })
    df = pd.DataFrame(rows)
    ds = ISICDataset(df, image_size=384, is_train=True, use_metadata=True,
                     use_segmentation_mask=False, mask_dir=None,
                     aug_cfg=config.get("augmentation", {}).get("train", {}))
    sample = ds[0]
    assert sample["image"].shape == (3, 384, 384), f"Shape: {sample['image'].shape}"
    assert "metadata" in sample
    assert sample["metadata"].shape == (13,)

test_dataset()


# ============================================================
# 9. TTADataset
# ============================================================
@test("tta_dataset")
def test_tta():
    from data import TTADataset
    tmpdir = tempfile.mkdtemp()
    from PIL import Image
    rows = []
    for i in range(2):
        img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
        path = os.path.join(tmpdir, f"tta_{i}.jpg")
        img.save(path)
        rows.append({
            "image": f"tta_{i}",
            "image_path": path,
            "label": i,
            "age_norm": 0.5,
            "sex_idx": 0,
            "site_idx": 5,
        })
    df = pd.DataFrame(rows)
    ds = TTADataset(df, image_size=384, use_metadata=True,
                    use_segmentation_mask=False, mask_dir=None)
    sample = ds[0]
    assert sample["images"].shape == (8, 3, 384, 384), f"Shape: {sample['images'].shape}"
    assert "metadata" in sample

test_tta()


# ============================================================
# 10. Full training step (forward + backward + optimizer + EMA)
# ============================================================
@test("full_training_step_mps")
def test_full_step():
    from model import build_model, get_layerwise_lr_groups
    from losses import build_loss
    from utils import EMA, WarmupCosineScheduler, clip_grad_norm
    
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    ema = EMA(model, decay=0.9995)
    
    lr_groups = get_layerwise_lr_groups(model, base_lr=1e-4)
    optimizer = torch.optim.AdamW(lr_groups, weight_decay=1e-5)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=80)
    
    # Forward
    x = torch.randn(BS, 4, IMG_SIZE, IMG_SIZE, device=device)
    meta = torch.randn(BS, META_DIM, device=device)
    labels = torch.randint(0, NUM_CLASSES, (BS,), device=device)
    
    logits = model(x, metadata=meta)["logits"]
    loss = criterion(logits, labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    gnorm = clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()
    ema.update()
    scheduler.step()
    
    print(f"    loss={loss.item():.4f}, grad_norm={gnorm:.4f}")

test_full_step()


# ============================================================
# 11. AMP (not available on MPS, check graceful handling)
# ============================================================
@test("amp_graceful_on_mps")
def test_amp():
    from model import build_model
    model = build_model(config).to(device)
    x = torch.randn(BS, 4, IMG_SIZE, IMG_SIZE, device=device)
    # MPS may or may not support autocast; either way it shouldn't crash
    try:
        with torch.cuda.amp.autocast(enabled=False):
            out = model(x)
        assert out["logits"].shape == (BS, NUM_CLASSES)
    except RuntimeError:
        # If autocast fails on MPS, that's expected — we disable it
        pass

test_amp()


# ============================================================
# 12. Checkpoint save/load
# ============================================================
@test("checkpoint_save_load")
def test_checkpoint():
    from model import build_model
    from utils import EMA, save_checkpoint, load_checkpoint
    model = build_model(config).to("cpu")  # CPU for checkpoint test
    ema = EMA(model, decay=0.9995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    save_checkpoint(model, optimizer, scheduler, ema, epoch=5, metric=0.85, path=path, config=config)
    assert os.path.exists(path)
    
    model2 = build_model(config)
    ema2 = EMA(model2, decay=0.9995)
    ckpt = load_checkpoint(path, model2, ema=ema2)
    assert ckpt["epoch"] == 5
    assert ckpt["best_metric"] == 0.85
    os.unlink(path)

test_checkpoint()


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print("  VERIFICATION SUMMARY")
print(f"{'='*50}")
passed = sum(1 for v in results.values() if v == "PASS")
total = len(results)
for name, status in results.items():
    icon = "✅" if status == "PASS" else "❌"
    print(f"  {icon} {name}: {status}")
print(f"\n  {passed}/{total} tests passed")

if passed < total:
    sys.exit(1)
