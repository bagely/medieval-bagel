import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def preprocess_imgs(style, content, reference):
  output_size = (min(style.size[1], content.size[1]),
                 min(style.size[0], content.size[0]))
  loader = transforms.Compose([transforms.Resize(output_size),
                               transforms.ToTensor()])
  style_tensor = Variable(loader(style))
  content_tensor = Variable(loader(content))
  if reference is not None:
    reference_tensor = Variable(loader(reference)).unsqueeze(0)
  else:
    reference_tensor = None
  return style_tensor.unsqueeze(0), content_tensor.unsqueeze(0), reference_tensor

class ContentLoss(nn.Module):
  def __init__(self, gt, weight):
    super(ContentLoss, self).__init__()
    self.gt = gt.detach() * weight
    self.weight = weight
  
  def forward(self, x):
    self.loss = nn.MSELoss()(x * self.weight, self.gt)
    return x
  
  def backward(self, retain_graph=True):
    self.loss.backward(retain_graph=retain_graph)
    return self.loss

def get_gram_matrix(x):
  N, C, W, H = x.shape
  G = x.view(N * C, W * H)
  return torch.mm(G, G.t()).div(N * C * W * H)

class StyleLoss(nn.Module):
  def __init__(self, gt, weight):
    super(StyleLoss, self).__init__()
    self.gt = gt.detach() * weight
    self.weight = weight
  
  def forward(self, x):
    x_copy = x.clone()
    gram_matrix = get_gram_matrix(x).mul_(self.weight)
    self.loss = nn.MSELoss()(gram_matrix, self.gt)
    return x_copy

  def backward(self, retain_graph=True):
    self.loss.backward(retain_graph=retain_graph)
    return self.loss

class Normalization(nn.Module):
  def forward(self, img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    if torch.cuda.is_available():
      mean = mean.to("cuda")
      std = std.to("cuda")
    return (img - mean) / std

vgg = models.vgg19(pretrained=True).features
if torch.cuda.is_available():
  vgg = vgg.to("cuda")
vgg.eval()

default_content_layers = ["conv4_2"]
default_style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
default_ref_layers = ["conv4_2"]

def get_model(content, style, reference,
              content_layers, style_layers, ref_layers,
              content_weight, style_weight, ref_weight):
  vgg_copy = copy.deepcopy(vgg)
  model = nn.Sequential()
  normalization = Normalization()
  if torch.cuda.is_available():
    model = model.to("cuda")
    normalization = normalization.to("cuda")
  model.add_module("normalization", normalization)
  
  level = 1
  i = 1
  count = len(content_layers) + len(style_layers) + len(ref_layers)

  content_losses = []
  style_losses = []
  ref_losses = []

  for layer in list(vgg_copy):
    if isinstance(layer, nn.MaxPool2d):
      name = "maxpool_" + str(level)
      level += 1
      i = 1
    elif isinstance(layer, nn.Conv2d):
      name = "conv{}_{}".format(str(level), str(i))
    else:
      name = "relu{}_{}".format(str(level), str(i))
      layer = nn.ReLU(inplace=False)
      i += 1
    model.add_module(name, layer)
    if name in content_layers:
      count -= 1
      content_target = model(content).clone()
      content_loss = ContentLoss(content_target, content_weight)
      model.add_module("content_loss_" + name, content_loss)
      content_losses.append(content_loss)
    if name in style_layers:
      count -= 1
      style_target = model(style).clone()
      weight = style_weight / layer.out_channels
      style_target_gram = get_gram_matrix(style_target)
      style_loss = StyleLoss(style_target_gram, weight)
      model.add_module("style_loss_" + name, style_loss)
      style_losses.append(style_loss)
    if reference is not None and name in ref_layers:
      count -= 1
      ref_target = model(reference).clone()
      ref_loss = ContentLoss(ref_target, ref_weight)
      model.add_module("reference_loss_" + name, ref_loss)
      ref_losses.append(ref_loss)
    if count == 0:
      break
  return model, content_losses, style_losses, ref_losses

def style_transfer(content, style, reference,
                   content_weight, style_weight, ref_weight,
                   content_layers=default_content_layers,
                   style_layers=default_style_layers,
                   ref_layers=default_ref_layers,
                   white_noise_input=False,
                   iteration=500):
  model_set = get_model(content, style, reference,
                        content_layers, style_layers, ref_layers,
                        content_weight, style_weight, ref_weight)
  model, content_losses, style_losses, ref_losses = model_set
  input = torch.randn(content.shape) if white_noise_input else content.clone()
  if torch.cuda.is_available():
    input = input.to("cuda")

  optimizer = optim.LBFGS([input.requires_grad_()])

  i = [0, float("inf")]
  while i[0] < iteration:
    def closure():
      input.data.clamp_(0, 1)
      optimizer.zero_grad()
      model(input)
      style_loss_val = 0
      content_loss_val = 0
      ref_loss_val = 0
      for style_loss in style_losses:
        style_loss_val += style_loss.loss
      for content_loss in content_losses:
        content_loss_val += content_loss.loss
      for ref_loss in ref_losses:
        ref_loss_val += ref_loss.loss
      i[0] += 1
      if i[0] % 100 == 0:
        print("Iteration: {}, Style loss: {}, "
              "Content loss: {}, "
              "Reference loss: {}".format(i[0],
                                          style_loss_val,
                                          content_loss_val,
                                          ref_loss_val))
      loss = style_loss_val + content_loss_val + ref_loss_val
      i[1] = loss.item()
      loss.backward()

      return loss
    optimizer.step(closure)

  input.data.clamp_(0, 1)

  return transforms.ToPILImage()(input.data[0].cpu())
