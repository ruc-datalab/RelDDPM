import sys
sys.path.append('../')
from ddpm import modules, diffusion, train
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import time
import data_utils as du
import itertools
from ddpm.resample import create_named_schedule_sampler
import os

def data_preprocessing(raw_data, label, save_dir=None):
    data_wrapper = du.DataWrapper()
    label_wrapper = du.DataWrapper()
    data_wrapper.fit(raw_data)
    label_wrapper.fit(raw_data[[label]])

    if save_dir is not None:
        du.save_pickle(data=data_wrapper, path=os.path.join(save_dir, 'data_wrapper.pkl'))
        du.save_pickle(data=label_wrapper, path=os.path.join(save_dir, 'label_wrapper.pkl'))
    return data_wrapper, label_wrapper

def set_anneal_lr(opt, init_lr, step, all_steps):
	frac_done = step / all_steps
	lr = init_lr * (1 - frac_done)
	for param_group in opt.param_groups:
		param_group["lr"] = lr

def diffuser_training(train_x, save_path, device, d_hidden=[512, 1024, 1024, 512], num_timesteps=1000, epochs=30000, lr=0.0018, drop_out=0.0, bs=4096):
	train_x = torch.from_numpy(train_x).float()
	model = modules.MLPDiffusion(train_x.shape[1], d_hidden, drop_out)
	model.to(device)
	print("Model Initialization")

	diff_model= diffusion.GaussianDiffusion(train_x.shape[1], model, device=device, num_timesteps=num_timesteps)
	diff_model.to(device)
	diff_model.train()
	print("Diffusion Initialization")

	ds = [train_x]
	dl = du.prepare_fast_dataloader(ds, batch_size = bs, shuffle = True)

	trainer = train.Trainer(diff_model, dl, lr, 0.0, epochs, save_path=None, device=device)
	train_sta = time.time()
	trainer.run_loop()
	train_end = time.time()
	print(f'training time: {train_end-train_sta}')

	diff_model.to(torch.device('cpu'))
	diff_model.variables_to_device(torch.device('cpu'))
	diff_model.eval()
	torch.save(diff_model, save_path)
	
def controller_training(train_x, train_y, diffuser, save_path, device, n_classes, lr=0.001, d_hidden=[512, 512], steps=10000, drop_out=0.0, bs=1024):
	train_x = torch.from_numpy(train_x).float()
	train_y = torch.from_numpy(train_y).float()
	
	model = modules.MLPClassifier(d_in=train_x.shape[1], d_layers=d_hidden, num_classes=n_classes, dropout=drop_out, t_in=True)
	ds = [train_x, train_y]
	dl = du.prepare_fast_dataloader(ds, batch_size = bs, shuffle = True)
	schedule_sampler = create_named_schedule_sampler("uniform", diffuser.num_timesteps)

	model.train()
	model.to(device)
	diffuser.to(device)
	diffuser.variables_to_device(device)

	opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00001)
	sta = time.time()
	for step in range(steps):
		x, y = next(dl)
		x = x.to(device)
		y = y.to(device)
		if n_classes == 2:
			y = y.squeeze()
		else:
			y = y.long()
			
		t, _ = schedule_sampler.sample(len(y), device)

		xt = diffuser.gaussian_q_sample(x, t)
		logits = model(xt, t)
		
		if n_classes > 2:
			loss = F.cross_entropy(logits, y)
		else:
			logits = logits.squeeze()
			loss = F.binary_cross_entropy(logits, y)
		
		opt.zero_grad()
		loss.backward()
		opt.step()
		
		set_anneal_lr(opt, lr, step, steps)
		
		if (step+1) % 100 == 0 or step == 0:
			print(f'Step {step+1}/{steps} : Loss {loss.data}')   
	end = time.time()

	model.to(torch.device('cpu'))
	model.eval()

	torch.save(model, save_path)
