import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import face
import os


class ConvBN (nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,relu=False):
        super(ConvBN,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = relu
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x,inplace=True)
        return x


class SSHContext (nn.Module):
    def __init__(self, channels, Xchannels=256):
        super(SSHContext, self).__init__()

        self.conv1 = nn.Conv2d(channels,Xchannels,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(channels,Xchannels//2,kernel_size=3,dilation=2,stride=1,padding=2)
        self.conv2_1 = nn.Conv2d(Xchannels//2,Xchannels//2,kernel_size=3,stride=1,padding=1)
        self.conv2_2 = nn.Conv2d(Xchannels//2,Xchannels//2,kernel_size=3,dilation=2,stride=1,padding=2)
        self.conv2_2_1 = nn.Conv2d(Xchannels//2,Xchannels//2,kernel_size=3,stride=1,padding=1)
        

    def forward(self, x):
        x1 = F.relu(self.conv1(x),inplace=True)
        x2 = F.relu(self.conv2(x),inplace=True)
        x2_1 = F.relu(self.conv2_1(x2),inplace=True)
        x2_2 = F.relu(self.conv2_2(x2),inplace=True)
        x2_2 = F.relu(self.conv2_2_1(x2_2),inplace=True)

        return torch.cat([x1,x2_1,x2_2],1)

class ContextTexture (nn.Module):
    """docstring for ContextTexture """
    def __init__(self, **channels):
        super(ContextTexture , self).__init__()
        self.up_conv = nn.Conv2d(channels['up'],channels['main'],kernel_size=1)
        self.main_conv = nn.Conv2d(channels['main'],channels['main'],kernel_size=1)
        

    def forward(self,up,main):
        up = self.up_conv(up)
        main = self.main_conv(main)
        _,_,H,W = main.size()
        res = F.upsample(up,scale_factor=2,mode='bilinear') 
        if res.size(2) != main.size(2) or res.size(3) != main.size(3):
            res = res[:,:,0:H,0:W]
        res = res + main
        return res

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = F.relu(self.bn2(self.conv2(out)),inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out,inplace=True)
        return out


class SFD(nn.Module):
    def __init__(self, block, num_blocks, phase , num_classes, size):
        super(SFD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBoxLayer(size,size,stride = [4,8,16,32,64,128])
        self.priors = None
        self.priorbox_head = PriorBoxLayer(size,size,stride = [8,16,32,64,128,128])
        self.priors_head = None
        self.priorbox_body = PriorBoxLayer(size,size,stride = [16,32,64,128,128,128])
        self.priors_body = None

        self.size = size
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)#256
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)#512
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)#1024
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)#2048
        self.layer5 = nn.Sequential(                                        #512
            *[nn.Conv2d(2048, 512, kernel_size=1,),                         #256
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512, kernel_size=3,padding=1,stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)]
            )
        self.layer6 = nn.Sequential(
            *[nn.Conv2d(512, 128, kernel_size=1,),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3,padding=1,stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)]
            )

        self.conv3_ct_py = ContextTexture(up=512,main=256)
        self.conv4_ct_py = ContextTexture(up=1024,main=512)
        self.conv5_ct_py = ContextTexture(up=2048,main=1024)
        
        self.latlayer_fc = nn.Conv2d(2048,2048,kernel_size=1)
        self.latlayer_c6 = nn.Conv2d(512,512,kernel_size=1)
        self.latlayer_c7 = nn.Conv2d(256,256,kernel_size=1)

        self.smooth_c3 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.smooth_c4 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.smooth_c5 = nn.Conv2d(1024,1024,kernel_size=3,padding=1)


        self.conv2_SSH = SSHContext(256,256)
        self.conv3_SSH = SSHContext(512,256)
        self.conv4_SSH = SSHContext(1024,256)
        self.conv5_SSH = SSHContext(2048,256)
        self.conv6_SSH = SSHContext(512,256)
        self.conv7_SSH = SSHContext(256,256)
        
        self.SSHchannels = [512,512,512,512,512,512]
        loc = []
        conf = []
        for i in range(6):
            loc.append(nn.Conv2d(self.SSHchannels[i],4,kernel_size=3,stride=1,padding=1))
            conf.append(nn.Conv2d(self.SSHchannels[i],4,kernel_size=3,stride=1,padding=1))

        self.face_loc = nn.ModuleList(loc)
        self.face_conf = nn.ModuleList(conf)

        head_loc = []
        head_conf = []
        for i in range(5):
            head_loc.append(nn.Conv2d(self.SSHchannels[i+1],4,kernel_size=3,stride=1,padding=1))
            head_conf.append(nn.Conv2d(self.SSHchannels[i+1],2,kernel_size=3,stride=1,padding=1))

        self.head_loc = nn.ModuleList(head_loc)
        self.head_conf = nn.ModuleList(head_conf)

        '''body_loc = []
        body_conf = []
        for i in range(4):
            body_loc.append(nn.Conv2d(self.SSHchannels[i+2],4,kernel_size=3,stride=1,padding=1))
            body_conf.append(nn.Conv2d(self.SSHchannels[i+2],2,kernel_size=3,stride=1,padding=1))

        self.body_loc = nn.ModuleList(body_loc)
        self.body_conf = nn.ModuleList(body_conf)'''


        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 750, 0.05, 0.3)



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)



    def forward(self, x):
        # Bottom-up

        sources = list()
        loc = list()
        conf = list()
        head_loc = list()
        head_conf = list()
        body_conf = list()
        body_loc = list()

        c1 = F.relu(self.bn1(self.conv1(x)),inplace=True)
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)    #S4
        c3 = self.layer2(c2)    #S8
        c4 = self.layer3(c3)    #S16
        c5 = self.layer4(c4)    #S32
        c6 = self.layer5(c5)   #S64
        c7 = self.layer6(c6)   #S128

        c5_lat = self.latlayer_fc(c5)
        c6_lat = self.latlayer_c6(c6)
        c7_lat = self.latlayer_c7(c7)

        c4_fuse = self.conv5_ct_py(c5_lat,c4)
        c3_fuse = self.conv4_ct_py(c4_fuse,c3)
        c2_fuse = self.conv3_ct_py(c3_fuse,c2)
        
        c2_fuse = self.smooth_c3(c2_fuse)
        c3_fuse = self.smooth_c4(c3_fuse)
        c4_fuse = self.smooth_c5(c4_fuse)
        

        c2_fuse = self.conv2_SSH(c2_fuse)
        sources.append(c2_fuse)
        c3_fuse = self.conv3_SSH(c3_fuse)
        sources.append(c3_fuse)
        c4_fuse = self.conv4_SSH(c4_fuse)
        sources.append(c4_fuse)
        c5_lat = self.conv5_SSH(c5_lat)
        sources.append(c5_lat)
        c6_lat = self.conv6_SSH(c6_lat)
        sources.append(c6_lat)
        c7_lat = self.conv7_SSH(c7_lat)
        sources.append(c7_lat)


        prior_boxs = []
        prior_head_boxes = []
        prior_body_boxes = []
        for idx, f_layer in enumerate(sources):
            #print('source size:',sources[idx].size())
            prior_boxs.append(self.priorbox.forward(idx,f_layer.shape[3],f_layer.shape[2]))
            if idx > 0:
                prior_head_boxes.append(self.priorbox_head.forward(idx-1,f_layer.shape[3],f_layer.shape[2]))
            #if idx > 1:
            #    prior_body_boxes.append(self.priorbox_body.forward(idx-2,f_layer.shape[3],f_layer.shape[2]))
        self.priors = Variable(torch.cat([p for p in prior_boxs],0),volatile=True)
        self.priors_head = Variable(torch.cat([p for p in prior_head_boxes],0),volatile=True)
        #self.priors_body = Variable(torch.cat([p for p in prior_body_boxes],0),volatile=True)


        for idx, (x, l, c) in enumerate(zip(sources, self.face_loc, self.face_conf)):
            if idx==0:
                tmp_conf = c(x)
                a,b,c,pos_conf = tmp_conf.chunk(4,1)
                neg_conf = torch.cat([a,b,c],1)
                max_conf,_ = neg_conf.max(1)
                max_conf = max_conf.view_as(pos_conf)
                conf.append(torch.cat([max_conf,pos_conf],1).permute(0,2,3,1).contiguous())
            else:
                tmp_conf = c(x)
                neg_conf,a,b,c = tmp_conf.chunk(4,1)
                pos_conf = torch.cat([a,b,c],1)
                max_conf,_ = pos_conf.max(1)
                max_conf = max_conf.view_as(neg_conf)
                conf.append(torch.cat([neg_conf,max_conf],1).permute(0,2,3,1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())


        
        for idx, (x, l, c) in enumerate(zip(sources[1:], self.head_loc, self.head_conf)):
            head_loc.append(l(x).permute(0,2,3,1).contiguous())
            head_conf.append(c(x).permute(0,2,3,1).contiguous())


        #for idx, (x, l, c) in enumerate(zip(sources[2:], self.body_loc, self.body_conf)):
        #    body_loc.append(l(x).permute(0,2,3,1).contiguous())
        #    body_conf.append(c(x).permute(0,2,3,1).contiguous())

            

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        

        head_loc = torch.cat([o.view(o.size(0), -1) for o in head_loc], 1)
        head_conf = torch.cat([o.view(o.size(0), -1) for o in head_conf], 1)
        #body_loc = torch.cat([o.view(o.size(0), -1) for o in body_loc], 1)
        #body_conf = torch.cat([o.view(o.size(0), -1) for o in body_conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, 2)),                         # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, 2),
                self.priors,
                head_loc.view(head_loc.size(0),-1,4),
                head_conf.view(head_conf.size(0),-1,2),
                self.priors_head
            )
        return output


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            pretrained_model = torch.load(base_file,map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            pretrained_model = {k : v for k, v in pretrained_model.items() if k in model_dict}
            model_dict.update(pretrained_model)
            self.load_state_dict(model_dict)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def build_sfd(phase, size=640, num_classes=2):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 640:
        print("Error: Sorry only 640 is supported currently!")
        return
    return SFD(Bottleneck, [3,4,6,3], phase , num_classes, size)
