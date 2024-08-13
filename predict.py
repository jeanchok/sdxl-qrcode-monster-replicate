import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from cog import BasePredictor, Input, Path
import tempfile

class Predictor(BasePredictor):
    def setup(self):
        self.controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sdxl_qrcode_monster")
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        qr_image: Path = Input(description="QR code image to use as control"),
        num_inference_steps: int = Input(description="Number of inference steps", default=30),
        guidance_scale: float = Input(description="Guidance scale", default=7.5),
    ) -> Path:
        image = self.pipeline(
            prompt,
            image=qr_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        output_path = Path(tempfile.mkdtemp()) / "output.png"
        image.save(output_path)
        return output_path