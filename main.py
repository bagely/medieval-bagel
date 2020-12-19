from PIL import Image
from torchvision import transforms
import torch
import imageio
import cv2
from transfer import style_transfer, preprocess_imgs

style_name = "img/1989087.jpg"
content_name = "img/content.jpg"
style_img = Image.open(style_name)
style_weight = 18000
content_weight = 1
ref_weight = 150
mode = "jpg" # or "gif" or "content representation" or "style representation"
do_gif = mode == "gif"
output_file = "output.gif" if do_gif else "output.jpg"

if do_gif:
  content = imageio.mimread(content_name)
  content_frames = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in content]
  outputs = []
  last_frame = None
  for i, frame in enumerate(content_frames):
    print("{}/{}".format(i, len(content_frames)))
    cv2.imwrite("hey.png", frame)
    content_img = Image.open("hey.png")
    style, content, reference = preprocess_imgs(style_img, content_img, last_frame)
    if torch.cuda.is_available():
      style = style.to('cuda')
      content = content.to('cuda')
      if reference is not None:
        reference = reference.to('cuda')
    output = style_transfer(content, style, reference,
                            content_weight, style_weight, ref_weight)
    outputs.append(output)
    last_frame = output
  
  with imageio.get_writer(output_file, mode='I', duration=1 / 30) as writer:
    for output in outputs:
      output.save("hey.png")
      im = imageio.imread("hey.png")
      writer.append_data(im)
    writer.close()

if mode == "jpg":
  content_img = Image.open(content_name)
  style, content, reference = preprocess_imgs(style_img, content_img, None)
  if torch.cuda.is_available():
    style = style.to('cuda')
    content = content.to('cuda')
  image = style_transfer(content, style, reference, content_weight, style_weight, ref_weight)
  image.save(output_file)

if mode == "content representation":
  content_img = Image.open(content_name)
  style, content, reference = preprocess_imgs(content_img, content_img, None)
  if torch.cuda.is_available():
    style = style.to('cuda')
    content = content.to('cuda')
  for i in ["conv1_2", "conv2_2", "conv3_2", "conv4_2"]:
    image = style_transfer(content, style, reference, 1, 0, 0,
                           content_layers=[i], style_layers=[], ref_layers=[],
                           white_noise_input=True)
    image.save("content_rep_{}.jpg".format(i))

if mode == "style representation":
  style, content, reference = preprocess_imgs(style_img, style_img, None)
  if torch.cuda.is_available():
    style = style.to('cuda')
    content = content.to('cuda')
  layers = []
  for i in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]:
    layers.append(i)
    image = style_transfer(content, style, reference, 0, 10000000, 0,
                           content_layers=[], style_layers=layers, ref_layers=[],
                           white_noise_input=True)
    image.save("style_rep_{}.jpg".format(i))
