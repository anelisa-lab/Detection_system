"""Settings taken from the GROUP11 research guide (Sections 3.3, 5 and 6)."""

IMG_SIZE = 224                     # ResNet50 input size (Section 5)
CLASS_NAMES = ["early", "mid", "late"]

# Gestational-stage bins on head circumference (mm). The guide says "clinically
# informed thresholds" without the numbers; these defaults are the HC values that
# Hadlock's HC->GA formula maps to ~24 and ~32 weeks. Override with --thresholds.
HC_THRESHOLDS_MM = (220.0, 290.0)

# Classification head (Section 5): 256-unit dense layer, 50% dropout, softmax.
HEAD_UNITS = 256
DROPOUT = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 10           # monitors val_loss

# Splits (Section 5): 80/20 train/test, then 20% of train held for validation.
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2
SEED = 42

LAST_CONV_LAYER = "conv5_block3_out"   # Grad-CAM target in Keras ResNet50
