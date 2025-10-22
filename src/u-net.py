import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(EncoderBlock,self).__init__()
    self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size =3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU(inplace= True)

    self.conv2= nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu2 = nn.ReLU(inplace= True)

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self,x):
    x = self.relu1(self.bn1(self.conv1(x)))
    x = self.relu2(self.bn2(self.conv2(x)))

    skip =x
    x = self.pool(x)
    return x, skip
class DecoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DecoderBlock,self).__init__()

    self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    #after concat

    self.conv1 = nn.Conv2d(out_channels *2, out_channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu2 = nn.ReLU(inplace= True)

  def forward(self, x, skip):
    x= self.upconv(x) # upsampling
    x = torch.cat([x,skip],dim=1)
    x= self.relu1(self.bn1(self.conv1(x)))
    x= self.relu2(self.bn2(self.conv2(x)))
    return x
x = torch.randn(1, 3, 256, 256)   # batch=1, channels=3, image=256x256
enc = EncoderBlock(3, 64)
down, skip = enc(x)

print("Downsampled:", down.shape)  # (1, 64, 128, 128)
print("Skip:", skip.shape)         # (1, 64, 256, 256)

x = torch.randn(1, 128, 64, 64)   # coming from deeper layer
skip = torch.randn(1, 64, 128, 128)  # encoder skip connection

dec =  DecoderBlock(128, 64)
out = dec(x, skip)

print("Output shape:", out.shape)  # (1, 64, 128, 128)
class UNet(nn.Module):
  def __init__(self, in_channels = 3, num_classes=5):
    super(UNet, self).__init__()

    self.enc1 = EncoderBlock(in_channels,64)
    self.enc2 = EncoderBlock(64,128)
    self.enc3 = EncoderBlock(128,256)
    self.enc4 = EncoderBlock(256, 512)

    self.bottleneck = nn.Sequential(
        nn.Conv2d(512,1024, kernel_size=3, padding=1),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024,1024, kernel_size= 3, padding=1),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace=True)
    )

    self.dec4 = DecoderBlock(1024,512)
    self.dec3 = DecoderBlock(512,256)
    self.dec2 = DecoderBlock(256,128)
    self.dec1 = DecoderBlock(128,64)

    self.final_conv = nn.Conv2d(64,num_classes,kernel_size=1)

  def forward(self,x):
    x1, skip1 = self.enc1(x)
    x2, skip2 = self.enc2(x1)
    x3, skip3 = self.enc3(x2)
    x4, skip4 = self.enc4(x3)
    x = self.bottleneck(x4)

    x = self.dec4(x,skip4)
    x = self.dec3(x,skip3)
    x = self.dec2(x,skip2)
    x = self.dec1(x,skip1)

    out = self.final_conv(x)

    return out


model = UNet(in_channels=3, num_classes=5)  # VOCSegmentation has 21 classes
x = torch.randn(1, 3, 256, 256)
out = model(x)
print("Output shape:", out.shape)


import torch.optim as optim
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr = 1e-4)

# dummy batch (like from DataLoader)
images = torch.randn(4, 3, 256, 256)      # batch of 4 images
masks = torch.randint(0, 5, (4, 256, 256))  # batch of 4 masks with values in [0, 20]

# forward
outputs = model(images)

# compute loss
loss = criterion(outputs, masks)

# backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())


import torch
import torch.nn as nn
import torch.optim as optim

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=5).to(device)  # SipakMed has 5 classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for images, masks in train_loader: # it is in spikemedl_classification.ipynb
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset) # # it is in spikemedl_classification.ipynb

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
