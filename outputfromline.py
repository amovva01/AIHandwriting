from data import IAMDataset
from model import HTR
from fastai.vision.all import show_image
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np



processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

inputs = Image.open("imagetoread.png").convert("RGB")

pixel_values = processor(inputs, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
output = generated_text

