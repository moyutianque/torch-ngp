from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from torchvision import transforms
import torch

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
	"stabilityai/stable-diffusion-2-1", 
	use_auth_token=True
)

vae = pipe.vae
# import cv2
# import numpy as np
# image = cv2.imread('./51.png', cv2.IMREAD_UNCHANGED)
# image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
# init_image = image.astype(np.float32) / 255 

init_image = Image.open('./51.png')

train_transforms = transforms.Compose(
	[
		# transforms.Resize(768, interpolation=transforms.InterpolationMode.BILINEAR),
		# transforms.CenterCrop(768),
		transforms.ToTensor(),
		# transforms.Normalize([0.5], [0.5]),
	]
)

def preprocess_train(pil_im):
	init_image = pil_im.convert("RGB") 
	# pixel_values =[transforms.ToTensor()(init_image)] # also work
	pixel_values = [train_transforms(init_image)]
	print(pixel_values[0].shape)
	pixel_values = torch.stack(pixel_values)
	print(pixel_values.shape)
	pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
	return pixel_values

inputs = preprocess_train(init_image)
vae = vae.to('cuda')

vae = vae.eval()
with torch.no_grad():
	# latents = vae.encode(inputs.cuda()).latent_dist.sample()
	distribution = vae.encode(inputs.cuda())
	latents = distribution.latent_dist.mode()
	print(latents.shape)
	# latents = latents * vae.config.scaling_factor
	preds = vae.decode(latents).sample

trans2pil = transforms.ToPILImage()
ori_img = trans2pil(inputs[0])
pred_img = trans2pil(preds[0])

ori_img.save('ori.png')
pred_img.save('pred.png')