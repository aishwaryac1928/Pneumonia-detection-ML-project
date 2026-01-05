# 🔧 DISK USAGE FIX GUIDE

## ⚠️ Problem
Running `run_pipeline.py` causes 100% disk usage and system shutdown because:

1. **Continuous Disk I/O**: ImageDataGenerator reads thousands of images from disk constantly
2. **Multiple Models Training**: Original script trains 2 models sequentially (ResNet50 + EfficientNetB0)
3. **Heavy Logging**: Frequent checkpoints and CSV logging write to disk constantly
4. **Large Dataset**: Processing thousands of X-ray images
5. **TensorFlow Cache**: TF writes large temporary files during training
6. **No Memory Management**: No garbage collection or memory clearing

## ✅ Solutions Provided

### Option 1: Safe Pipeline (RECOMMENDED)
**File**: `run_pipeline_safe.py`

**Features**:
- ✅ System resource monitoring
- ✅ Disk space checks before starting
- ✅ Train ONE model at a time
- ✅ Automatic cleanup of temp files
- ✅ Graceful interruption handling
- ✅ Progress monitoring

**Usage**:
```bash
python run_pipeline_safe.py
```

**What it does**:
1. Checks available disk space (needs 10GB+)
2. Checks available RAM (needs 4GB+)
3. Lets you choose which model to train
4. Monitors disk I/O during training
5. Cleans up temporary files automatically

### Option 2: Optimized Training Script
**File**: `train_optimized.py`

**Optimizations**:
- ✅ Reduced batch size: 32 → 16 (50% less memory)
- ✅ Reduced epochs: 20+10 → 10+5 (50% less time)
- ✅ Smaller dense layers (less parameters)
- ✅ Memory cleanup after each phase
- ✅ Less frequent disk writes
- ✅ Non-interactive matplotlib backend
- ✅ GPU memory growth enabled

**Usage**:
```bash
python train_optimized.py
```

## 📋 Step-by-Step Instructions

### Before Running:

1. **Check Disk Space**:
   - Open File Explorer
   - Right-click C: drive → Properties
   - Ensure you have **at least 10GB free**

2. **Close Other Programs**:
   - Close browsers with many tabs
   - Close any video editing/heavy software
   - Close unnecessary background apps

3. **Install Required Package**:
   ```bash
   pip install psutil
   ```

### Running the Safe Pipeline:

```bash
# Step 1: Run the safe pipeline
python run_pipeline_safe.py

# Step 2: Choose option
# Option 1: Train ResNet50 only (RECOMMENDED for first run)
# Option 2: Train EfficientNetB0 only
# Option 3: Evaluate existing models

# Step 3: Monitor in Task Manager
# Press Ctrl+Shift+Esc
# Go to Performance tab → Disk
# If disk hits 95%+, press Ctrl+C to stop safely
```

### Running Direct Training:

```bash
# Train one model with optimizations
python train_optimized.py

# Choose which model (ResNet50 recommended first)
```

## 🛡️ Safety Features

### Automatic Monitoring:
- Checks disk space before starting
- Monitors disk I/O during training
- Warns if disk usage is too high
- Safe interruption with Ctrl+C

### Memory Management:
- Garbage collection after each phase
- Keras session clearing
- Smaller batch sizes
- Limited workers

### Disk Protection:
- Stops if disk usage > 85%
- Less frequent model saves
- Automatic temp file cleanup
- Reduced logging

## 🔍 Monitoring Your System

### During Training, Watch:

1. **Task Manager** (Ctrl+Shift+Esc):
   - Performance → Disk: Should stay < 90%
   - Memory: Should stay < 85%
   - CPU: Will be high (normal)

2. **Signs of Trouble**:
   - ⚠️ Disk at 100% for > 1 minute
   - ⚠️ System becoming unresponsive
   - ⚠️ Disk usage warning appears
   
3. **If Trouble Occurs**:
   - Press Ctrl+C to stop safely
   - Model checkpoints are saved
   - Resume later with less load

## 📊 Expected Performance

### Original Script:
- ❌ 100% disk usage
- ❌ System crashes
- ❌ 2-6 hours (if it doesn't crash)

### Optimized Scripts:
- ✅ 50-80% disk usage
- ✅ Stable operation
- ✅ 2-4 hours per model
- ✅ Can be interrupted safely

## 🎯 Recommended Workflow

### Day 1:
```bash
# Run safe pipeline for ResNet50
python run_pipeline_safe.py
# Choose option 1 (ResNet50 only)
# Let it train (2-4 hours)
```

### Day 2:
```bash
# Run safe pipeline for EfficientNetB0
python run_pipeline_safe.py
# Choose option 2 (EfficientNetB0 only)
# Let it train (2-4 hours)
```

### After Training:
```bash
# Compare models
python utils.py

# Run demo
python app.py
```

## 🆘 Troubleshooting

### Problem: Still getting high disk usage
**Solution**:
1. Reduce batch size further in `train_optimized.py`:
   ```python
   batch_size=8  # Change from 16 to 8
   ```
2. Reduce epochs:
   ```python
   epochs=5  # Change from 10
   fine_tune_epochs=3  # Change from 5
   ```

### Problem: Out of memory error
**Solution**:
1. Close ALL other applications
2. Reduce batch size to 8 or even 4
3. Use only one model at a time

### Problem: Training too slow
**Solution**:
- This is normal! Each model takes 2-4 hours
- Use faster storage (SSD) if possible
- Don't interrupt - let it complete

### Problem: PC still shuts down
**Solution**:
1. Check if overheating (clean dust from fans)
2. Ensure adequate power supply
3. Update GPU drivers
4. Run in safe mode with even smaller batch size (4)

## 📝 What's Different?

| Feature | Original | Optimized |
|---------|----------|-----------|
| Batch Size | 32 | 16 (50% less) |
| Epochs | 20+10 | 10+5 (50% less) |
| Models | Both at once | One at a time |
| Monitoring | None | Full monitoring |
| Cleanup | Manual | Automatic |
| Disk Writes | Frequent | Optimized |
| Memory | No management | Active cleanup |

## 💡 Pro Tips

1. **Run overnight**: Start training before bed, let it run
2. **One model at a time**: Don't try to rush
3. **Monitor first 30 min**: Watch disk usage, if stable, you're good
4. **Save power**: Use high performance power plan
5. **Close browser**: Chrome/Edge can use lots of RAM
6. **Check temps**: Ensure PC has good cooling

## ✅ Success Checklist

Before starting training:
- [ ] At least 10GB free disk space
- [ ] At least 4GB free RAM
- [ ] All other programs closed
- [ ] psutil installed (`pip install psutil`)
- [ ] Chose ONE model to train first
- [ ] Task Manager open to monitor

Good luck! 🚀
