import torch

class VGG(torch.nn.Module):

    def __init__(self):
        super(VGG, self).__init__()

        self.config_lst = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        self.model_lst = self.vgg(self.config_lst, 3)

    def vgg(self, cfg, i, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = torch.nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                torch.nn.ReLU(inplace=True), conv7, torch.nn.ReLU(inplace=True)]
        return layers

    def forward(self, x, model_lst):
        for i, layer in enumerate(model_lst):
            if i > 0 and i < len(model_lst)-1:
                x = layer(x)
        return x

model = VGG()
print(model)

# torch.save(model.state_dict(), "vgg_weights.pth")
# print("SAVED")

# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
# model.eval()

# import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

# from PIL import Image
# from torchvision import transforms
# input_image = Image.open(filename)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

# # Read the categories
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
# # Show top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#     print(categories[top5_catid[i]], top5_prob[i].item())