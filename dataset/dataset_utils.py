import torchvision.transforms as transforms

def tensor_to_image(tensor_image):
  to_pil_image = Compose([Resize(img_size), ToPILImage()])
  image = to_pil_image(tensor_image)
  return image