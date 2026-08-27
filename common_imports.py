import os
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from types import SimpleNamespace

from timm import create_model

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
