
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import pdb
from collections import OrderedDict

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


resnext_50_32x4d = nn.Sequential( # Sequential,
	nn.Conv2d(3,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
	nn.Sequential( # Sequential,
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(64,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(128),
						nn.ReLU(),
						nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(128),
						nn.ReLU(),
					),
					nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(256),
				),
				nn.Sequential( # Sequential,
					nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(256),
				),
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(128),
						nn.ReLU(),
						nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(128),
						nn.ReLU(),
					),
					nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(256),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(128),
						nn.ReLU(),
						nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(128),
						nn.ReLU(),
					),
					nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(256),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
	),
	nn.Sequential( # Sequential,
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
						nn.ReLU(),
						nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(256),
						nn.ReLU(),
					),
					nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(512),
				),
				nn.Sequential( # Sequential,
					nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(512),
				),
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
						nn.ReLU(),
						nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(256),
						nn.ReLU(),
					),
					nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(512),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
						nn.ReLU(),
						nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(256),
						nn.ReLU(),
					),
					nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(512),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
						nn.ReLU(),
						nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(256),
						nn.ReLU(),
					),
					nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(512),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
	),
	nn.Sequential( # Sequential,
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
						nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
					),
					nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(1024),
				),
				nn.Sequential( # Sequential,
					nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(1024),
				),
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
						nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
					),
					nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(1024),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
						nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
					),
					nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(1024),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
						nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
					),
					nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(1024),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
						nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
					),
					nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(1024),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
						nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(512),
						nn.ReLU(),
					),
					nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(1024),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
	),
	nn.Sequential( # Sequential,
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
						nn.ReLU(),
						nn.Conv2d(1024,1024,(3, 3),(2, 2),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(1024),
						nn.ReLU(),
					),
					nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(2048),
				),
				nn.Sequential( # Sequential,
					nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(2048),
				),
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
						nn.ReLU(),
						nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(1024),
						nn.ReLU(),
					),
					nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(2048),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
		nn.Sequential( # Sequential,
			LambdaMap(lambda x: x, # ConcatTable,
				nn.Sequential( # Sequential,
					nn.Sequential( # Sequential,
						nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
						nn.ReLU(),
						nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
						nn.BatchNorm2d(1024),
						nn.ReLU(),
					),
					nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
					nn.BatchNorm2d(2048),
				),
				Lambda(lambda x: x), # Identity,
			),
			LambdaReduce(lambda x,y: x+y), # CAddTable,
			nn.ReLU(),
		),
	),
#	nn.AvgPool2d((7, 7),(1, 1)),
#	Lambda(lambda x: x.view(x.size(0),-1)), # View,
)

class resnext_car_multitask(nn.Module):
    def __init__(self, cropsize=224, resnext_model=None, class_num=1, test=False):
        super(resnext_car_multitask, self).__init__()
        self.resnext_car_multitask=resnext_50_32x4d
        self.classifier = [] 
        self.class_num = class_num
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)

        pool_size = int(cropsize/32.0)
        self.ave_pool = nn.AvgPool2d((pool_size, pool_size),(1, 1))
        self.max_pool = nn.MaxPool2d((pool_size,pool_size),(1,1))
        embed_size = 2048


        for i in range(class_num):

            self.classifier.append(nn.Linear(embed_size,2))

        self.classifier = nn.ModuleList(self.classifier)
        for params in self.parameters():
               params.requires_grad = False  






        self.cropsize = cropsize
        for params in self.parameters():
            if params.ndimension()>1:
                torch.nn.init.xavier_uniform(params)
            else:
                torch.nn.init.normal(params)
        if False:
            print('loading model')
            params = torch.load(resnext_model)
            keys = params.keys()
            # pop 1000 fc for loading models
            keys1 = list(keys)
            pdb.set_trace()
            if test:
             new_state_dict = OrderedDict()
             for k,v in params.items():
                word = k.split('.')  
                l = len(word[0])
                name = k[l+1:]
                new_state_dict[name] = v
             self.resnext_car_multitask.load_state_dict(params)
            else:
                params.pop(keys1[-1])
                params.pop(keys1[-2])
                self.resnext_car_multitask.load_state_dict(params)
    def forward(self, x):
        x = x.view(-1, 3, self.cropsize, self.cropsize)
        x = self.resnext_car_multitask(x)
        outputs = []
        outputs2 = []
        plot = False
        if plot:

            feature = self.fc_0(x)
            feature = feature.data
            feature = feature.cpu()
            feature0 = feature.numpy()

            feature = self.fc_1(x)
            feature = feature.data
            feature = feature.cpu()
            feature1 = feature.numpy()

            feature = self.fc_2(x)
            feature = feature.data
            feature = feature.cpu()
            feature2 = feature.numpy()
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x,2)        
        

        for i in range(self.class_num):
            outputs.append(self.classifier[i](x))



        return outputs, outputs2

def resnext50_fg_car(pretrained=False, model_dir='', class_num=1, test=False, **kwargs):
    if pretrained:
#        model_dict = torch.load(model_dir)
        model = resnext_car_multitask(resnext_model=model_dir, class_num=class_num, test=test,  **kwargs)
        params = torch.load(model_dir)
        keys = params.keys()
        keys1 = list(keys)
        if  test:
             print('load imagent model')
             params.pop(keys1[-1])
             params.pop(keys1[-2])
             new_state_dict = OrderedDict()
             for k,v in params.items():                
                name = 'resnext_car_multitask.'+k
                print(name) 
                new_state_dict[name] = v
             state = model.state_dict()
             state.update(new_state_dict)
             model.load_state_dict(state)
        else:
             print('load test model')
             new_state_dict = OrderedDict()
             for k,v in params.items():                
                name = k[7:]

 

                new_state_dict[name] = v
             state = model.state_dict()
             state.update(new_state_dict)
             model.load_state_dict(state)
           
 
    else:
        model = resnext_car_multitask(resnext_model=None, class_num=class_num, test=test,  **kwargs)
    return model

