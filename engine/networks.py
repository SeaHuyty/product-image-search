import numpy as np
import torch
import torch.nn.functional as F 
from PIL import Image 
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder:
    def __init__(self, model_name, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)


    def _to_numpy(self, feats):
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        feats = F.normalize(feats.float(), dim=-1)

        return feats.cpu().numpy()


    @torch.no_grad()
    def encode_text(self, texts):
        if  isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        return self._to_numpy(self.model.get_text_features(**inputs))

    
    @torch.no_grad()
    def encode_images(self, images):
        images = [img.convert("RGB") for img in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        return self._to_numpy(self.model.get_image_features(**inputs))


if __name__ == "__main__":
    from PIL import Image
    enc = CLIPEncoder("openai/clip-vit-base-patch32")
    t = enc.encode_text(["red dress"])
    i = enc.encode_images([Image.open("engine/data/images/15970.jpg")])

    print(t.shape, i.shape, (t ** 2).sum(), (t @ i.T))