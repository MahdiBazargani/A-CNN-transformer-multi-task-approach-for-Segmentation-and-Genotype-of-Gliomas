from models.unet.unet_parts import *



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels,32)
        self.down1 = DoubleConv(32, 64,stride=2)
        self.down2 = DoubleConv(64, 128,stride=2)
        self.down3 = DoubleConv(128, 256,stride=2)
        self.down4 = DoubleConv(256, 320,stride=2)
        self.down5= DoubleConv(320, 320,stride=2)

        self.up1 =  Up(320, 320)
        self.up2 =  Up(320, 256)
        self.up3 =  Up(256, 128)
        self.up4 =  Up(128, 64)
        self.up5 =  Up(64, 32)
        self.outc = OutConv(32, n_classes)
        self.outc1 = OutConv2(64,n_classes,2)
        self.outc2 = OutConv2(128,n_classes,4)
        self.outc3 = OutConv2(256,n_classes,8)
        self.outc4 = OutConv2(320,n_classes,16)
        
        
    def forward(self, x):
        logits=[]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        logits.append(self.outc4(x))
        x = self.up2(x, x4)
        logits.append(self.outc3(x))
        x = self.up3(x, x3)
        logits.append(self.outc2(x))
        x = self.up4(x, x2)
        logits.append(self.outc1(x))
        x = self.up5(x, x1)
        logits.append(self.outc(x))
        return logits
    