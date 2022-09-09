#! /usr/local/bin/python3
# @original author Yanfei Tang
'''
 Largely modified by Lingxiao Wang and Shuzhe Shi
 Reference:
 	Rethinking the ill-posedness of the spectral function reconstruction 
 	- why is it fundamentally hard and how Artificial Neural Networks can help
 	by Shuzhe Shi, Lingxiao Wang, and Kai Zhou
  	arXiv:2201.02564[hep-ph] (https://arxiv.org/abs/2201.02564)
'''

import sys
import os
import numpy as np
from numpy import exp, log, sin, cos, sqrt, pi, dot, array, diag, loadtxt, savetxt, logspace
from numpy import linspace, zeros, real, ones, abs, prod, linalg, sum, isnan, append, insert, random
import time
import pickle
#import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from numpy.linalg import inv

def F_inv_g(g):
	if (g>-2): f=g
	else: f = -log(-g)
	dif = 1;
	attemp = 0;
	while attemp<50 and abs(dif) > 1e-10:
		F   = 1+exp(-f)
		dif = g-F*f
		dF  = f-F*f+F
		f  += dif/dF
		attemp += 1
	return f

def BW(m, g, w):
    return (4*g*w) / ((m**2+g**2-w**2)**2 + (2*g*w)**2)

def sfunction(w):
	return ones(len(w))

class Maxent(object):
	def __init__(self, folder, numW = 500, wmax = 20, numK = 100, kmax = 20, numS = 600, smax = 30, tol = 1e-8, alphamin = -4, alphamax = 0, numAlpha = 100, entropy_type = "SJ", defaultModel = "sfunction", minimizer = "Bryan", rank=0, mu=0.1, maxIteration=1e4, CreateNew=False, noise=1e-4):
		self.start_time = time.time()
		self.folder = folder
		self.numW = numW
		self.wmax = wmax
		self.numK = numK
		self.kmax = kmax
		self.numS = numS
		self.smax = smax
		self.tol = tol
		self.alphamin = alphamin
		self.alphamax = alphamax
		self.numAlpha = numAlpha
		self.minimizer = minimizer
		self.entropy_type = entropy_type
		self.rank = rank
		self.mu = mu
		self.maxIteration = maxIteration
		self.allSpecFs, self.allProbs = [], []
		self.k_assigned = False
		if CreateNew:
			self.createKernel()
			self.createSKernel()
			self.createData(noise)
		else:
			self.readfile()
			self.createKernel()
			self.createSKernel()
			self.compUXiV()
		if defaultModel == 'sfunction':
			self.specF= sfunction(self.w)
			self.defaultM = self.specF
		elif defaultModel == 'input':
			data = loadtxt(sys.argv[3])
			self.specF, self.defaultM = data[:, 1], data[:, 1]
		else:
			print("Usage: Maxent(filename, numW, wmin, wmax, defaultModel, tol, std, alphamin, alphamax, numAlpha)")
			print("defaultModel must 'g' or 's'!")
			sys.exit(0)
		print("Use --- %s seconds --- before optimization..." %(time.time() - self.start_time))
			  
	def readfile(self):
		"""
		read data from file:
		for D(k) in real sapce
		file format:
		#   1   2   3
		1  k1  D1  dD1
		2  k2  D2  dD2
		"""
		self.k, self.D_obs, self.D_std = [], [], []
		with open('./%s/True_D.txt'%self.folder, 'r') as inputfile:
			for line in inputfile:
				a = line.split('\t')
				self.k.append(float(a[0]))
				self.D_obs.append(float(a[1]))
				self.D_std.append(float(a[2]))
		inputfile.close()
		self.k = array(self.k)
		self.D_obs = array(self.D_obs)
		self.D_std = array(self.D_std)
		self.k_assigned = True
		self.numK = len(self.k)
 
	def createKernel(self):
		""" 
			The Convolution Kernel:
			K = w / pi / (wn^2 + w^2) 
		"""
		self.dw = self.wmax/self.numW
		self.w  = linspace(self.dw, self.wmax, self.numW)
		if not self.k_assigned:
			self.dk = self.kmax/self.numK
			self.k  = linspace(self.dk, self.kmax, self.numK)
		self.K = zeros([self.numK, self.numW], dtype = 'float64')
		for n in range(self.numK):
			for m in range(self.numW):
				self.K[n][m] = (self.w[m]/(self.w[m]**2 + self.k[n]**2))/pi
		self.K_full = zeros([self.numW, self.numW], dtype = 'float64')
		for n in range(self.numW):
			for m in range(self.numW):
				self.K_full[n][m] = (self.w[m]/(self.w[m]**2 + self.w[n]**2))/pi
 
	def createSKernel(self):
		""" 
		Kernel of generalized Fourier Transformation
		"""
		self.s = linspace((self.smax)/(self.numW), self.smax, self.numS)
		self.KS_cos = zeros([self.numS, self.numW], dtype = 'float64')
		self.KS_sin = zeros([self.numS, self.numW], dtype = 'float64')
		for m in range(self.numS):
			s = self.s[m]
			for n in range(self.numW):
				#self.KS_cos[m][n] = cos(log(self.w[n]) * s) * self.dw
				#self.KS_sin[m][n] = sin(log(self.w[n]) * s) * self.dw
				w_p = self.w[n]+0.5*self.dw
				w_m = self.w[n]-0.5*self.dw
				ws_p = log(w_p) * s
				ws_m = log(w_m) * s
				self.KS_cos[m][n] = (w_p*(cos(ws_p)+s*sin(ws_p)) - w_m*(cos(ws_m)+s*sin(ws_m)))/(1+s*s)
				self.KS_sin[m][n] = (w_p*(sin(ws_p)-s*cos(ws_p)) - w_m*(sin(ws_m)-s*cos(ws_m)))/(1+s*s)
			ws = log(0.5*self.dw) * s
			wl = log(self.wmax) * s
			comfac_s = 0.5*self.dw/(1+s*s)
			comfac_l = (self.wmax)**2/((1+s*s)*(self.wmax+0.5*self.dw))
			self.KS_cos[m][0]  += (cos(ws)+s*sin(ws))*comfac_s
			self.KS_sin[m][0]  += (sin(ws)-s*cos(ws))*comfac_s
			self.KS_cos[m][-1] += (cos(wl)-s*sin(wl))*comfac_l
			self.KS_sin[m][-1] += (sin(wl)+s*cos(wl))*comfac_l

	def createData(self, epsilon):
		print('Warning! Creating new data table!')
		rho = 0.8*BW(2.0, 0.5, self.w) + 1.0*BW(5.0, 0.5, self.w)
		#rho = BW(3.0, 0.5, self.w)
		#rho = BW(2.5, 0.1, self.w)
		D0  = self.D_cal(rho)
		#dD  = ones(self.numW)
		#dD  = epsilon*ones(self.numW)
		dD  = epsilon*D0*self.k/self.dk
		rho_out = open('./%s/True_rho.txt'%self.folder,'w')
		for i in range(self.numW):
			rho_out.write('%f\t%f\n'%(self.w[i],rho[i]))
		rho_out.close()
		D_out = open('./%s/True_D.txt'%self.folder,'w')
		for i in range(self.numK):
			#D_out.write('%f\t%e\t%e\n'%(self.k[i],D0[i]+dD[i]*random.normal(0,1,1),dD[i]))
			D_out.write('%f\t%e\t%e\n'%(self.k[i],D0[i],dD[i]))
		D_out.close()
		self.print_s(rho,'./%s/True_rho_D_s.txt'%(self.folder))

	def print_s(self,rhow,filename):
		'''
		Print rho and D in generalized momentum space
		'''
		Dk = dot(self.K_full, rhow) * self.dw
		[rho_c, rho_s] = self.S_cal(rhow)
		[D0_c,  D0_s]  = self.S_cal(Dk)
		s_out = open(filename,'w')
		for i in range(self.numS):
			s_out.write('%f\t%f\t%f\t%f\t%f\n'%(self.s[i],rho_c[i],rho_s[i],D0_c[i],D0_s[i]))
		s_out.close()

	def print_D_and_s_for_ext_file(self,filein,fileD,fileout):
		'''
		Read in rho(omega) and D(k) reconstructed from external files and 
		then performs the generalized Fourier transformation
		'''
		rho_in = loadtxt('./{}/{}'.format(self.folder,filein), delimiter='\t', unpack=True)
		rhow = rho_in[1]
		Dk = self.D_cal(rhow)
		D_out = open('./{}/{}'.format(self.folder,fileD),'w')
		for i in range(self.numK):
			D_out.write('%f\t%e\n'%(self.k[i],Dk[i]))
		D_out.close()
		self.print_s(rhow,'./{}/{}'.format(self.folder,fileout))

	def compUXiV(self):
		"""
		Perform the sigular value decomposition to the kernel matrix K.
		"""
		fout = open('SVD_result.dat', 'w')
		self.U, self.Xi, self.Vt = linalg.svd(self.K.transpose(), full_matrices = 0)
		xi_str='\t'.join(['%e'%i for i in self.Xi])
		fout.write('N_omega:\t%d\n'%self.numW)
		fout.write('N_obsvd:\t%d\n'%self.numK)
		fout.write('EigenValues:\n'+xi_str+'\n')
		fout.write('V_transpose:\n')
		for j in range(self.numK):
			v_str='\t'.join(['%e'%i for i in self.Vt[j]])
			fout.write(v_str+'\n')
		fout.write('U:\n')
		for j in range(self.numW):
			u_str='\t'.join(['%e'%i for i in self.U[j]])
			fout.write(u_str+'\n')
		if self.rank<1:
			self.rank = np.linalg.matrix_rank(self.K.transpose())
			print('Using:rank=%d'%self.rank)
		self.Xi = diag(self.Xi[:self.rank])
		self.Vt = self.Vt[:self.rank, :]
		self.V  = self.Vt.transpose()
		self.U  = self.U[:, :self.rank]
		self.Ut = self.U.transpose()
		self.M  = dot(dot(self.Xi, self.Vt), diag(1.0/self.D_std/self.D_std))
		self.M  = dot(self.M, self.V)
		self.M  = dot(self.M, self.Xi)
		self.M  = self.dw*self.M
		
	def D_cal(self, specF):
		"""
		From the spectral function to find out the D: D = K * rho
		"""
		return dot(self.K, specF) * self.dw

	def S_cal(self, f):
		"""
		Perform generalized Fourier transformation 
		"""
		return [dot(self.KS_cos, f) / sqrt(pi),
				dot(self.KS_sin, f) / sqrt(pi)]

	def chiSquare(self, specF):
		"""
		chi^2 = |(D_obs - K * A) / sigma|^2
		"""
		delta = (self.D_obs - self.D_cal(specF))/self.D_std
		return real(sum((delta)**2))

	def entropy(self, specF):
		"""
		Entropy term in difference regulators; 
			'SJ'   : 'Shannon–Jaynes' used in MEM, see Eq.(50)
			'TK'   : 'Tikanov', see Eq.(49)
			'L2SP' : 'L2regularizer with SoftPlus activation', see Eq.(92)
		"""
		if self.entropy_type == 'SJ':
			result = sum( specF - self.defaultM - (specF * log(1e-12+abs((specF)/self.defaultM))) ) 
		elif self.entropy_type == 'TK':
			result = - 0.5*sum((specF-self.defaultM)**2) 
		elif self.entropy_type == 'L2SP':
			result = -0.5*sum((log(abs(exp(specF/self.defaultM) - 1.)))**2)
		return result * self.alpha * self.dw

	def objective(self, specF):
		"""
		Q = 1/2 \\chi^2 - S
		considering the standard deviation or not.
		"""
		return 0.5*self.chiSquare(specF) - self.entropy(specF)

	def getSpecF(self, printrecord=True):
		"""
		using SLSQP aka sequential least square quadratic programing or Bryan's method (R. K. Bryan, Eur. Biophys. J., 18 (1990) 165) to minimize the objective function to get the spectral function depending on alpha.
		"""
		if self.minimizer != 'Bryan': 
			print('use Bryan minimizer!, Stopping')
			exit()

		iteration = 0
		c_coef = zeros(self.rank)
		fa = zeros(self.numW)
		ga = zeros(self.numW)
		self.specF = self.defaultM
		Qold = self.objective(self.defaultM)
		file = './{}/{}_{}_{}.rec.dat'.format(self.folder,self.minimizer, self.entropy_type, self.rank)
		#record = open(file, 'w')
		while True:
			if iteration > self.maxIteration:
				print("Exceeds maximum iteration in Levenberg-Marquart algorithms, exits. Make tolerance smaller.")
				break
			iteration += 1
			Delta = (self.D_obs - self.D_cal(self.specF))/self.D_std/self.D_std
			# g is the function that can be decomposed into the singular space 
			dchi2_dc = dot(dot(self.Xi, self.Vt), Delta) 
			# F_zero is the funtion supposed to be zero
			F_zero = dchi2_dc - self.alpha * c_coef

			#drho_dg = \frac{d rho(g)}{d g}; 
			if self.entropy_type == 'SJ':
				drho_dg = self.specF
			elif self.entropy_type == 'TK':
				drho_dg = ones(self.numW)
			elif self.entropy_type == 'L2SP':
				ffd = 1 + exp(-fa)
				drho_dg = real(self.defaultM / ffd / (ffd-fa*exp(-fa)))
			T = dot(dot(self.Ut, diag(drho_dg)), self.U)
			LHS = (self.alpha + self.mu) * diag(ones(self.rank)) + dot(self.M, T)
			d_c_coef = dot(inv(LHS), F_zero)
			c_coef = c_coef + d_c_coef
			N_print = 1
			# c_coef updated, then compute g_a and rho_a
			ga = dot(self.U, c_coef)
			if self.entropy_type == 'SJ':
				self.specF = real(self.defaultM * exp(ga))
			elif self.entropy_type == 'TK':
				self.specF = self.defaultM + ga
			elif self.entropy_type == 'L2SP':
				for i in range(self.numW): fa[i] = F_inv_g(ga[i])
				self.specF = real(self.defaultM * log(1+exp(fa)))
			self.chi2 = self.chiSquare(self.specF)
			self.entp = self.entropy(self.specF)
			Qnew = self.objective(self.specF)
			if (iteration%N_print==0) and printrecord:
				rec = '{}\t{}\t{}'.format(iteration, self.chi2, self.entp)
				print(rec)
				coef_str = '\t'.join(['%.3f'%(c) for c in c_coef])
				#record.write(rec+'\t;\t'+coef_str+'\n')
			if abs(Qnew - Qold) < self.tol and iteration>4:
				print("{0} iterations in Levenberg-Marquart algorithms. Function evaluted: {1}, it exits properly.".format(iteration, Qnew))
				break
			Qold = Qnew
		#record.close()

	def print_rho_alpha(self):
		surfix = '%s_%s_%d_%.0e'%(self.minimizer,self.entropy_type,self.rank,self.alpha)
		'''
		plt.clf()
		plt.plot(self.w, self.specF, "b-", alpha = 0.8, label = "MEM")
		plt.xlabel(r"$\omega$")
		plt.ylabel(r"$\rho(\omega)$")
		plt.xlim([0,10])
		truth = loadtxt('./{}/True_rho.txt'.format(self.folder), delimiter='\t', unpack=True)
		plt.plot(truth[0], truth[1], "r-", alpha = 0.8, label = "Truth")
		ax = plt.gca()
		plt.text(0.7, 0.80, 'method : %s'%self.minimizer, fontsize=12, transform = ax.transAxes)
		plt.text(0.7, 0.73, 'rank = %d'%self.rank, fontsize=12, transform = ax.transAxes)
		plt.text(0.7, 0.66, '$\chi^2 = %.3e$'%(self.chi2), fontsize=12, transform = ax.transAxes)
		plt.text(0.7, 0.59, '$\mathcal{S}_{[%s]} = %.3e$'%(self.entropy_type,self.entp), fontsize=12, transform = ax.transAxes)
		plt.legend()
		plt.savefig("./{}/Rec_rho_{}.pdf".format(self.folder,surfix))
		'''
		result = open("./{}/Rec_rho_{}.txt".format(self.folder,surfix), 'w')
		for i in range(self.numW):
			result.write(str(self.w[i]) + '\t' + str(self.specF[i]) + "\n")
		result.close()
		Dk = self.D_cal(self.specF)
		result = open("./{}/Rec_D_{}.txt".format(self.folder,surfix), 'w')
		for i in range(self.numK):
			result.write(str(self.k[i]) + '\t' + str(Dk[i]) + "\n")
		result.close()
		self.print_s(self.specF, "./{}/Rec_rho_D_s_{}.txt".format(self.folder,surfix))

	def calProb(self):
		"""
		Compute the probablity for this paticular alpha. This probablity is not normalized.
		"""
		cov   = diag(self.D_std **2)
		mat_a = dot(self.K.transpose(), inv(cov))
		mat_a = dot(mat_a, self.K)
		vec_a = sqrt(abs(self.specF))
		imax, jmax = mat_a.shape
		mat_b = zeros((imax, jmax))
		for i in range(0, imax):
			for j in range(0, jmax):
				mat_b[i][j] = vec_a[i] * mat_a[i][j] * vec_a[j] 
		S = linalg.eigvalsh(mat_b)
		expo = exp(-self.objective(self.specF))
		produ = prod(self.alpha/(self.alpha+S))
		self.prob = sqrt( produ ) * expo/self.alpha
		if isnan(self.prob):
			self.prob = 0.0

	def getAllSpecFs(self):
		"""
		Compute all the spectral functions by looping all the alphas.
		"""
		self.alphas = logspace(self.alphamin, self.alphamax, self.numAlpha)

		# Use uniformed space integral for \int P(alpha)dalpha;
		self.dalpha = self.alphas[1:] - self.alphas[:-1]
		self.dalpha /= 2.0
		self.dalpha = insert(self.dalpha, 0, 0.0) + append(self.dalpha, 0.0)

		for self.alpha in self.alphas:
			self.getSpecF(printrecord=False)
			self.calProb()
			self.allSpecFs.append(self.specF)
			self.allProbs.append(self.prob)
			print("Finish alpha = %s.\n" %(self.alpha))

		self.allSpecFs = array(self.allSpecFs)
		self.allProbs = array(self.allProbs)
		if self.allProbs.sum()==0:
			self.allProbs = ones(self.numAlpha)
			self.allProbs = self.allProbs/sum(self.allProbs*self.dalpha)
		else:
			self.allProbs = self.allProbs/sum(self.allProbs*self.dalpha)

		self.aveSpecFs = dot(self.allSpecFs.transpose() * self.dalpha, self.allProbs)
		print("Optimization ends. Use --- %s seconds ---" %(time.time() - self.start_time))

	def saveObj(self):
		"""
		Save the result
		"""
		surfix = '{}_{}_{}'.format(self.minimizer,self.entropy_type,self.rank)

		result = open("./{}/Rec_rho_{}_Int.txt".format(self.folder,surfix), 'w')
		for i in range(self.numW):
			result.write(str(self.w[i]) + '\t' + str(self.aveSpecFs[i]) + "\n")
		result.close()

		Dk = self.D_cal(self.aveSpecFs);
		result = open("./{}/Rec_D_{}_Int.txt".format(self.folder,surfix), 'w')
		for i in range(self.numK):
			result.write(str(self.k[i]) + '\t' + str(Dk[i]) + "\n")
		result.close()

		self.print_s(self.aveSpecFs,"./{}/Rec_rho_D_s_{}_Int.txt".format(self.folder,surfix))
		result = open("./{}/Palpha_{}.txt".format(self.folder,surfix), 'w')
		for i in range(len(self.alphas)):
			result.write(str(self.alphas[i]) + '\t' + str(self.allProbs[i]) + "\n")
		result.close()

		chi2 =  self.chiSquare(self.aveSpecFs)
		print('chi-square = {}'.format(chi2))

def excute_model(directory, method, entp_type, rnk, mu=.1, maxIteration=1e4, alpha=1e-3):
	mu_update = mu
	max_ = maxIteration
	Model = Maxent(folder = directory, tol = 1e-8,
		minimizer = method, entropy_type=entp_type, rank=rnk, mu=mu_update, maxIteration=max_)
	Model.alpha = alpha
	Model.getSpecF()
	Model.print_rho_alpha()

def excute_full_model(directory, method, entp_type, rnk, mu=.1, maxIteration=1e4):
	mu_update = mu
	max_ = maxIteration
	Model = Maxent(folder = directory, tol = 1e-8, alphamin = -4, alphamax = 0, numAlpha = 300,\
		minimizer = method, entropy_type=entp_type, rank=rnk, mu=mu_update, maxIteration=max_)
	Model.getAllSpecFs()
	Model.saveObj()

def create_rho_D_s_for(directory):
	'''
	read in rho(omega) and D(k) reconstructed from NN and NN-P2P and 
	then performs the generalized Fourier transformation
	'''
	Model = Maxent(folder = directory)
	for ext in ['NN','NNP2P']:
		Model.print_D_and_s_for_ext_file('Rec_rho_%s.txt'%ext,'Rec_D_%s.txt'%ext,'Rec_rho_D_s_%s.txt'%ext)

if __name__ == "__main__":
	# generate SLSQP curve in Fig.3 left
	excute_model(directory='data/fig3', method='Bryan', entp_type='L2SP', rnk=100, mu=1e-5, alpha=1e-6, maxIteration=1e4)
	# generate MEM curves in Figure 4 and 5
	create_rho_D_s_for('data/fig4')
	create_rho_D_s_for('data/fig5')
	excute_full_model(directory='data/fig4', method='Bryan', entp_type='SJ', rnk=8, mu=.05)
	excute_full_model(directory='data/fig4', method='Bryan', entp_type='SJ', rnk=100, mu=.05)
	excute_full_model(directory='data/fig5', method='Bryan', entp_type='SJ', rnk=8, mu=.05)
	excute_full_model(directory='data/fig5', method='Bryan', entp_type='SJ', rnk=100, mu=.05)
	
