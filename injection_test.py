import os,sys,numpy as np,matplotlib.pyplot as plt
import pickle
def diff_func(q,A):
	return (-2.5*np.log10(2)-A)*q**2+A*q**3

def single_color_mag(mass,age,feh):
	bands = ['b','g','r']
	all_mags = []
	for band in bands:
		inFile = open('mag_coeffs_'+band,'r')
		all_coeffs = []
		for line in inFile:
			entry = line.split(' ')
			coeffs = []
			for ii in range(len(entry)):
				coeffs.append(float(entry[ii]))
			all_coeffs.append(coeffs)
		if type(mass)==int or type(mass)==float or type(mass)==np.float64:
				mass = np.array([mass])
				age = np.array([age])
				feh = np.array([feh])
		next_coeffs = np.zeros((len(all_coeffs),len(mass)))
		for ii in range(len(next_coeffs)):
			order = len(all_coeffs[ii])-1
			for jj in range(order+1):
				next_coeffs[ii]+=all_coeffs[ii][jj]*(mass**(order-jj))
		slope = next_coeffs[0]*feh+next_coeffs[1]
		iCept = next_coeffs[2]*feh+next_coeffs[3]
		mag = slope*age+iCept
		all_mags.append(mag)
	return all_mags

def binary_color_mag(mass1,mass_ratio,age,feh):
	bands = ['b','g','r']
	all_mags = []
	single_mags = single_color_mag(mass1,age,feh)
	for band_ii in range(len(bands)):
		inFile = open('diff_coeffs_'+bands[band_ii],'r')
		all_coeffs = []
		for line in inFile:
			entry = line.split(' ')
			coeffs = []
			for ii in range(len(entry)):
				coeffs.append(float(entry[ii]))
			all_coeffs.append(coeffs)
		if type(mass1)==int or type(mass1)==float or type(mass1)==np.float64:
			next_coeffs = np.zeros(len(all_coeffs))
		else:
			next_coeffs = np.zeros((len(all_coeffs),len(mass1)))
		for ii in range(len(next_coeffs)):
			order = len(all_coeffs[ii])-1
			for jj in range(order+1):
				next_coeffs[ii]+=all_coeffs[ii][jj]*(mass1**(order-jj))
		slope = next_coeffs[0]*feh+next_coeffs[1]
		iCept = next_coeffs[2]*feh+next_coeffs[3]
		A = slope*age+iCept
		magDiff = diff_func(mass_ratio,A)
		all_mags.append(single_mags[band_ii]+magDiff)
	return all_mags

def log_prior(theta):
    mass1,mass_ratio,feh,distance,age = theta
    if type(mass1)==float or type(mass1)==np.float64:
        if 0.08<mass1<1.8 and 0<=mass_ratio<=1 and -0.5<feh<0.2 and distance>0.1 and age>0.2 and age<6:
            return 0.0
        return -np.inf
    else:
        prior = np.zeros(len(mass1))
        for ii in range(len(mass1)):
            if 0.08<mass1[ii]<1.8 and 0<=mass_ratio[ii]<=1 and -0.5<feh[ii]<0.2 and distance[ii]>0.1 and age[ii]>0.2 and age[ii]<6:
                continue
            else:
                prior[ii] = -np.inf
        return prior

def llh_function(theta,apparent_mags,parallax,relative_flux_errors,par_err):
	mass1,mass_ratio,feh,distance,age = theta
	model_mags = binary_color_mag(mass1,mass_ratio,age,feh)
	model_b = model_mags[0]+5*np.log10(distance/10)
	model_g = model_mags[1]+5*np.log10(distance/10)
	model_r = model_mags[2]+5*np.log10(distance/10)
	b_errs = find_mag_errors(relative_flux_errors[0])
	g_errs = find_mag_errors(relative_flux_errors[1])
	r_errs = find_mag_errors(relative_flux_errors[2])
	[b_app,g_app,r_app] = apparent_mags
	model_par = 1./distance
	if type(g_app)==int or type(g_app)==float or type(g_app)==np.float64:
		b_err = b_errs[model_b>=b_app]
		g_err = g_errs[model_g>=g_app]
		r_err = r_errs[model_r>=r_app]
		log_likelihood = -0.5*(((b_app-model_b)/b_err)**2+((g_app-model_g)/g_err)**2+((r_app-model_r)/r_err)**2+((parallax-model_par)/par_err)**2)
	else:
		log_likelihood = np.zeros(len(model_g))
		for ii in range(len(g_app)):
			b_err = b_errs[model_b>=b_app[ii]]
			g_err = g_errs[model_g>=g_app[ii]]
			r_err = r_errs[model_r>=r_app[ii]]
			log_likelihood+=(-0.5*(((b_app[ii]-model_b)/b_err[ii])**2+((g_app[ii]-model_g)/g_err[ii])**2+((r_app[ii]-model_r)/r_err[ii])**2+((parallax[ii]-model_par)/par_err[ii])**2))
	for ii in range(len(log_likelihood)):
		if np.isnan(log_likelihood[ii]):
			log_likelihood[ii] = -np.inf
	return log_likelihood

def log_prob(theta,apparent_mags,parallax,relative_flux_errors,par_err):
    lp = log_prior(theta)
    if type(lp)==float or type(lp)==np.float64:
        if not np.isfinite(lp):
            return -np.inf
        return lp + llh_function(theta,apparent_mags,parallax,relative_flux_errors,par_err)
    else:
        prob = np.zeros(len(lp))
        llh = llh_function(theta,apparent_mags,parallax,relative_flux_errors,par_err)
        for ii in range(len(lp)):
            if not np.isfinite(lp[ii]):
                prob[ii] = -np.inf
            else:
                prob[ii] = lp[ii] + llh[ii]
        return prob

[isochrone_m1s,isochrone_qs,isochrone_ages,isochrone_fehs,[isochrone_bs,isochrone_gs,isochrone_rs]] = pickle.load(open('isochrone_grid.pkl', 'rb'))
nMass = 81
nFeH = 81
nAge = 81
test_m1s = np.linspace(0.08,1.9,nMass)
test_qs = np.linspace(0,1,nMass)
test_fehs = np.linspace(-1,0.5,nFeH)
test_ages = np.logspace(-1,1,nAge)
[test_M1s,test_Qs,test_FEHs,test_AGEs] = np.meshgrid(test_m1s,test_qs,test_fehs,test_ages)
test_m1s = test_M1s.flatten()
test_qs = test_Qs.flatten()
test_fehs = test_FEHs.flatten()
test_ages = test_AGEs.flatten()
[test_bs,test_gs,test_rs] = binary_color_mag(test_m1s,test_qs,test_ages,test_fehs)
for ii in range(len(isochrone_m1s)):
	diff = np.sqrt((isochrone_bs[ii]-test_bs)**2+(isochrone_gs[ii]-test_gs)**2+(isochrone_rs[ii]-test_rs)**2)
	minEl = diff.argmin()
	calculated_m1 = test_m1s[minEl]
	calculated_q = test_qs[minEl]
	calculated_feh = test_fehs[minEl]
	calculated_age = test_ages[minEl]
	print(isochrone_m1s[ii],calculated_m1,isochrone_qs[ii],calculated_q,isochrone_fehs[ii],calculated_feh,isochrone_ages[ii],calculated_age)