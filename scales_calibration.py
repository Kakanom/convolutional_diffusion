import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import argparse
import shutil
import sys
import os

from utils.noise_schedules import cosine_noise_schedule
from utils.data import get_dataset
from models import DDIM, MinimalUNet, MinimalResNet

from utils.idealscore import (
	ScheduledScoreMachine,
	LocalEquivBordersScoreModule,
	LocalEquivScoreModule,
	LocalScoreModule,
	IdealScoreModule
)

'''
THIS SCRIPT CALIBRATES THE EFFECTIVE LOCALITY SCALES OF THE NN MODELS FOR COMPARISON WITH ELS MACHINE
'''

def calibrate(
	kfilename='kfile',
	tld='./scales/',
	modelfile=None,
	dataset_name='mnist',
	scoremoduletype='bbELS',
	conditional=False,
	kernelsizes=None,
	scorebatchsize=8,
	nsamps=20,
	nsteps=20,
	nlabels=10,
	eval_mode='cos',
	cpu=False,
	maxsamps=100000
):
	"""
	Calibrate the effective locality scales of the NN models.

	Returns:
		A dictionary containing:
			- 'k_optimals': tensor of all kernel sizes chosen for each sample/time-step
			- 'median': tensor of the median kernel sizes across samples for each time-step
			- 'mode': tensor of the mode kernel sizes across samples for each time-step
	"""

	# --------------------
	# Dataset + Device
	# --------------------
	dataset, metadata = get_dataset(dataset_name, root='./data')
	if maxsamps < len(dataset):
		dataset = torch.utils.data.Subset(dataset, [i for i in range(maxsamps)])
	in_channels = metadata['num_channels']
	image_size = metadata['image_size']

	if cpu: 
		device = torch.device('cpu')
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# --------------------
	# Load Model
	# --------------------
	fname = os.path.join(tld, modelfile)
	model = torch.load(fname, map_location=device)
	model.eval()
	model.to(device)

	# --------------------
	# Score Modules
	# --------------------
	schedule = cosine_noise_schedule
	mods = []
	for kernel_size in kernelsizes:
		if scoremoduletype == 'ELS':
			mod = LocalEquivScoreModule(dataset,
						kernel_size=kernel_size,
						batch_size=scorebatchsize,
						image_size=image_size,
						channels=in_channels,
						schedule=schedule)
		elif scoremoduletype == 'bbELS':
			mod = LocalEquivBordersScoreModule(dataset,
						kernel_size=kernel_size,
						batch_size=scorebatchsize,
						image_size=image_size,
						channels=in_channels,
						schedule=schedule)
		elif scoremoduletype == 'LS':
			mod = LocalScoreModule(1, dataset,
						kernel_size=kernel_size,
						image_size=image_size,
						batch_size=len(dataset),
						schedule=schedule)
		else:
			raise ValueError(f"Unknown scoremoduletype: {scoremoduletype}")

		mod.to(device)
		mod.eval()
		mods.append(mod)

	# --------------------
	# Calibration Loop
	# --------------------
	nsteps = nsteps
	kcosine = torch.zeros(len(kernelsizes), nsteps, device=device)
	kdists = torch.zeros(len(kernelsizes), nsteps, device=device)
	k_optimals = torch.zeros(nsamps, nsteps, device=device)

	with torch.no_grad():
		for s in range(nsamps):
			# Optional label
			label = torch.randint(0, nlabels, (1,)) if conditional else None
			if label is not None:
				label = label.to(device)

			# Sample initial noise
			x = torch.randn((1, in_channels, image_size, image_size), device=device)

			# Reverse diffusion steps
			for i in range(nsteps, 0, -1):
				t = i * torch.ones(1, device=device) / nsteps
				beta_t = schedule(t)  
				if conditional:
					eps = model(t, x, label=label)
				else:
					eps = model(t, x, label=None)

				alpha_t = 1 - beta_t
				beta_t_prev = schedule(t - 1/nsteps)
				alpha_t_prev = 1 - beta_t_prev

				# Compute local score estimates
				k_estims = [m(t, x, label=label) for m in mods]

				# Update x
				x *= torch.sqrt(alpha_t_prev/alpha_t)[:, None, None, None]
				score_correction = (
					torch.sqrt(beta_t_prev[:, None, None, None]) - 
					torch.sqrt(alpha_t_prev/alpha_t)[:, None, None, None] * 
					torch.sqrt(beta_t[:, None, None, None])
				) * eps
				x += score_correction

				# Compare corrected_out to local estimates
				corrected_out = -eps / (beta_t**0.5)
				for j, ke in enumerate(k_estims):
					kdists[j, i-1] = torch.sqrt(torch.sum((corrected_out - ke)**2))
					kcosine[j, i-1] = (
						torch.sum(corrected_out * ke) /
						(torch.sqrt(torch.sum(corrected_out**2)) * torch.sqrt(torch.sum(ke**2)))
					)

				# Find best kernel index
				kmindist = kernelsizes[torch.argmin(kdists[:, i-1])]
				kmaxcos  = kernelsizes[torch.argmax(kcosine[:, i-1])]

				if eval_mode == 'l2_dist':
					k_optimals[s, i-1] = kmindist
				else:
					k_optimals[s, i-1] = kmaxcos

	# Compute final aggregated values
	k_median = torch.median(k_optimals, dim=0).values.type(torch.int)
	k_mode_vals = torch.mode(k_optimals, dim=0).values.type(torch.int)

	return {
		'k_optimals': k_optimals.cpu(),
		'median': k_median.cpu(),
		'mode': k_mode_vals.cpu()
	}


def main():
	parser = argparse.ArgumentParser(description='Calibrate')
	parser.add_argument('--kfilename', type=str, default='kfile')
	parser.add_argument('--tld', type=str, default='./scales/')
	parser.add_argument('--modelfile', type=str, default=None)
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--scoremoduletype', type=str, default='bbELS') 
	parser.add_argument('--conditional', action="store_true", default=False)
	parser.add_argument('--kernelsizes', type=int, nargs='*')
	parser.add_argument('--scorebatchsize', type=int, default=16)
	parser.add_argument('--nsamps', type=int, default=20)
	parser.add_argument('--nsteps', type=int, default=20)
	parser.add_argument('--nlabels', type=int, default=10)
	parser.add_argument('--eval_mode', type=str, default='cos')
	parser.add_argument('--cpu', action="store_true", default=False)
	parser.add_argument('--maxsamps', type=int, default=5000)

	args = parser.parse_args()

	# Run calibration
	results = calibrate(
		kfilename=args.kfilename,
		tld=args.tld,
		modelfile=args.modelfile,
		dataset_name=args.dataset,
		scoremoduletype=args.scoremoduletype,
		conditional=args.conditional,
		kernelsizes=args.kernelsizes,
		scorebatchsize=args.scorebatchsize,
		nsamps=args.nsamps,
		nsteps=args.nsteps,
		nlabels=args.nlabels,
		eval_mode=args.eval_mode,
		cpu=args.cpu,
		maxsamps=args.maxsamps
	)

	# Save the returned data
	torch.save(results['k_optimals'], f"{args.kfilename}_k_optimals.pt")
	torch.save(results['median'],     f"{args.kfilename}_median.pt")
	torch.save(results['mode'],       f"{args.kfilename}_mode.pt")


if __name__ == '__main__':
	main()
