# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:13:22 2021

@author: zhangbowen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchviz import make_dot


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
     
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        y = self.conv3(x)

        return y


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class AttentionZBW(nn.Module):
    
    def __init__(self):
        super(AttentionZBW, self).__init__()
        
        self.shallow = nn.Sequential(                    
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(),

            #nn.MaxPool2d(2),
        )

        self.non_local = nn.Sequential(
            NONLocalBlock2D(32),
        )
        
        # self.deep1 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        # )
        #
        # self.deep2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        # )
        #
        # self.deep3 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=8, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        # )
   
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )    
    
    def forward(self, x):
        height=x.shape[2]
        width=x.shape[3]

        shallow=self.shallow(x)
        
        upsample=self.non_local(shallow)
        
        # deep1=self.deep1(non_local)
        # deep2=self.deep2(non_local)
        # deep3=self.deep3(non_local)
        #
        # concat=torch.cat([deep1,deep2,deep3],dim=1)

        # upsample = F.interpolate(concat, size=(height, width), mode='bilinear', align_corners=True)

        last = self.last(upsample)

        return last


if __name__=='__main__':

    
    ##### SRCNN #####
#    model=SRCNN()
#    x=torch.randn(1,1,21,21)
#    y=model(x)
#    print('x:',x.shape)
#    print('y:',y.shape)
    
            
    
    ##### AttentionZBW #####
    model = AttentionZBW()
    x= torch.randn(1, 1, 100, 100)
    y = model(x)
    print('x:', x.shape)
    print('y:', y.shape)
        
    
    # network structure visualization
#    visualizer=make_dot(y) # method 1   
#    visualizer=make_dot(y, params=dict(model.named_parameters())) # method 2
#    visualizer=make_dot(y,params=dict(list(model.named_parameters())+[('x',x)])) # method 3
#    visualizer.format='pdf'
#    visualizer.render(r'./srcnn', view=True)
#    visualizer.render(r'./attentionzbw', view=True)
