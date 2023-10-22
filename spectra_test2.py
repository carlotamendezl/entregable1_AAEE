#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:19:39 2023

@author: carlota
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os



# ESPECTROS DE ESTRELLAS DE PRUEBA
# Leemos el archivo .dat como una DataFrame: 
    # 1era columna: Wavelength
    # 2a columna:   Flux

def read_dat(path, name, spaces = True, plotting = False):
        
        df1 = pd.read_csv(path, delim_whitespace=True, header=None, names=['Wavelength', 'Flux'])
        
        df1['Wavelength'] = pd.to_numeric(df1['Wavelength'], errors='coerce').astype(float)
        df1['Flux'] = pd.to_numeric(df1['Flux'], errors='coerce').astype(float)

        df1_cut = df1.iloc[39:4348] #Los otros elementos son 0
        
        wvl1 = df1_cut['Wavelength'].values.tolist()
        wvl1 = [float(i) for i in wvl1] #Porque los lee como string
        flux1 = df1_cut['Flux'].values.tolist()
        flux1 = [float(i) for i in flux1]
        
        if plotting == True:
            #Pintando el espectro
            xticks = np.arange(wvl1[0], wvl1[-1], 500) 
            
            plt.figure()
            plt.plot(wvl1, flux1)
            plt.xticks(xticks)
            
            plt.xlabel('Wavelength $[Å]$')
            plt.ylabel('Flux')
            plt.title('%s'%name)
            plt.grid()
            plt.tight_layout()
        
        return df1_cut

df1 = read_dat('Desktop/Atmosferas_estelares/entregable1/HD78558_prueba1.dat',\
               'starprob1', plotting=False)

df2 = read_dat('Desktop/Atmosferas_estelares/entregable1/HD161695_prueba2.dat',\
               'starprob2', plotting=False)
    
    
df1_prob = read_dat('Desktop/Atmosferas_estelares/entregable1/starprob1.dat',\
               'starprob1', plotting=True)

df2_prob = read_dat('Desktop/Atmosferas_estelares/entregable1/starprob2.dat',\
               'starprob2', plotting=True)

directory_path = 'Desktop/Atmosferas_estelares/entregable1/espectros_referencia'


def to_df(path):
    elements = os.listdir(path)
    files = [i for i in elements if i.endswith('.dat')]

    dic = {}
    for file in files:
        if file.endswith('.dat'):
            dic[file] = [] 
    
    
    for file in files:
        # Construct the absolute path of the file
        file_path = os.path.join(path, file)
        # Check if the file is indeed a file (not a directory)
        if os.path.isfile(file_path):
            if file.endswith('.dat'):
        
                # Open and read the file
                dic[file] = read_dat(file_path, file, plotting=False,spaces=True)
    
    return files, dic

files, refs_dict = to_df(directory_path)
  

Getiq = ['CaI', 'CaI', 'CaII', 'CaII', 'FeI', 'CH band', 'H$_\gamma$', 'SrII', 'H$_\delta$']
Gpos = [ 4226, 4260, 3969.7, 3934, 4045, 4300, 4340, 4077, 4102]

Aetiq = ['H$_\gamma$', r'H$\beta$','H$_\delta$', 'HeI', 'HeI', 'CaII', 'CaII', 'CaI', 'FeI', 'FeI', 'SrI', 'SiII', 'SiIII', 'MgII', 'SiIV', 'HeI']
Apos = [4340, 4861, 4102, 4026, 4471, 3969.7, 3934, 4226, 4271, 4383, 4077, 4128,  4552, 4481, 4089, 4121]
   

def multiple_plots(star_df, title, dic, files, n_col =6, spt=False, ratio=False):
    rows = len(dic)//n_col
    
    if len(dic)%n_col != 0:
        rows += 1
    
    columns = n_col
    
    fig, axs = plt.subplots(rows, columns, sharex=True, sharey=True)
    
    
    for i in range(len(dic)):
        if rows == 0:
            ax = axs[i % columns]
        
        if n_col==1:
            ax = axs[i]
            
        elif rows!=0:
            ax = axs[i//columns, i % columns]
        

        ax.plot(star_df['Wavelength'], star_df['Flux'], color='gray')
        ax.plot(dic[files[i]]['Wavelength'], dic[files[i]]['Flux'], alpha=.5, color='tab:blue')
        ax.axhline(1, color='k', ls='dashed', alpha=.5)

        #ax.set_xlim(3900, 5100)
        ax.set_ylim(0, 1.2)
        
        if i%columns == 0:
            ax.set_ylabel('Normalized flux')

        if i // columns == rows-1:
            ax.set_xlabel('Wavelength $[Å]$')
        

        if spt == 'G':
            for j in range(len(Gpos)):
                ax.axvline(Gpos[j], color='k', alpha=.6, ls='dashed')
                ax.text(Gpos[j] + 0.1, 1.05, str(Getiq[j]), fontsize=10, color='red')
            ax.set_xlim(min(Gpos)-10, max(Gpos)+10)
            
                    
        if spt == 'A':
            for j in range(len(Apos)):
                ax.axvline(Apos[j], color='k', alpha=.6, ls='dashed')
                
                if j == 13:
                    ax.text(Apos[j] + 5, 1.05, str(Aetiq[j]), fontsize=10, color='red')
                
                elif j ==4:
                    ax.text(Apos[j] - 20, 1.05, str(Aetiq[j]), fontsize=10, color='red')

                elif j==1:
                    ax.text(Apos[j] - 20, 1.05, str(Aetiq[j]), fontsize=10, color='red')

                elif j==2:
                    ax.text(Apos[j] + 0.1 , 1.05, str(Aetiq[j]), fontsize=10, color='red')
                    
                elif j==10:
                    ax.text(Apos[j] - 15, 1.05, str(Aetiq[j]), fontsize=10, color='red')
                
                elif j==14:
                    ax.text(Apos[j] - 10, 1.05, str(Aetiq[j]), fontsize=10, color='red')
                    
                else:
                    ax.text(Apos[j] + 0.1, 1.05, str(Aetiq[j]), fontsize=10, color='red')
                
            ax.set_xlim(min(Apos)-10, max(Apos)+15)    



        #ax.grid()

        ax.set_title('%s' %files[i])
        #xticks = np.arange(3900, 5100, 300) 
        #ax.set_xticks(xticks)

        fig.suptitle('%s' %title)
        plt.tight_layout()
    plt.show()

    
star_name = ['HD78558_prueba1', 'HD161695_prueba2']    
prob_name = ['starprob1', 'starprob2']    
   

#%%        
from astropy import units as u
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
from scipy.interpolate import splrep, BSpline


def envelope(df, star_index, wvl_short, wvl_long, plotting=False, resid=False, splines2=False):
    
    #Con SPECUTILS 
    flux = df['Flux'][445:].values.tolist()*u.Jy
    wvl = df['Wavelength'][445:].values.tolist()*u.nm

    spec = Spectrum1D(flux=flux, spectral_axis=wvl)

    g1_fit = fit_generic_continuum(spec)
    y_continuum_fitted = g1_fit(spec.spectral_axis)


    #Con SPLINES
    flux_new = ([])     #Short
    flux_new2 = ([])    #Long

    for i in range(len(wvl_short)):
        a = np.where(df['Wavelength']>wvl_short[i])[0][0]
        a = a+38
        flux_new.append(df['Flux'][a])
    
    for i in range(len(wvl_long)):
        b = np.where(df['Wavelength']>wvl_long[i])[0][0]
        b = b+38
        flux_new2.append(df['Flux'][b])
    
    
    tck = splrep(wvl_short, flux_new, xb=None, s=100)
    tck2 = splrep(wvl_long, flux_new2, xb=None, s=100)


    if plotting == True:
        # Set the default line width for all lines in the figure
        plt.rc('lines', linewidth=.8)  # Set the default line width for all lines

        #PROPOSED ENVELOPES
        fig_size, ratio, dpi = 30, 2, 150
        fig, ax = plt.subplots(figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),dpi=dpi,layout='constrained')
        
        ax.plot(df['Wavelength'], df['Flux'], color='gray', alpha=.5)
        ax.plot(spec.spectral_axis, y_continuum_fitted, color='b', ls=':', label='Specutils')
        ax.plot(df['Wavelength'][445:1739], BSpline(*tck)(df['Wavelength'][445:1739]), color='m', ls='dashed', label='Splines 1')
        ax.plot(df['Wavelength'], BSpline(*tck2)(df['Wavelength']), color='g', ls='-.', label='Splines 2')
        ax.plot(wvl_long, flux_new2, 'k.', label='Fitting dots')
        
        ax.set_xlabel('Wavelength $[Å]$')
        ax.set_ylabel('Flux')
        
        ax.grid(which="major",linewidth=np.sqrt(fig_size) * 0.015, color="black")
        
        plt.legend(loc='lower left', fontsize=8)
        plt.title('Spectrum envelope models for %s' %star_index) 


    
        #COMPAIRING ENVELOPES & CALCULATING RESIDUALS
        fig_size, ratio, dpi = 30, 2, 150
        fig, ax = plt.subplots(figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),dpi=dpi,layout='constrained')
        ax.set_title('Comparing envelope models')
        
        ax_in = fig.add_axes([0.55, 0.3, 0.4, 0.3])  #[x_length, y_length, x_width, y_width]
        
        if splines2 == True:
            norm2 = df['Flux']/BSpline(*tck2)(df1['Wavelength'])
            ax.plot(df['Wavelength'], norm2, color='g', alpha=.7, label='Splines 2')
            ax_in.plot(df['Wavelength'][445:1739], norm2[445:1739], color='g', alpha=.7, label='Splines 2')
            
        norm = df['Flux'][445:1739]/BSpline(*tck)(df['Wavelength'][445:1739])
        
        ax.plot(df['Wavelength'][445:1739], norm, color='m', alpha=.7, label='Splines 1')
        ax.axhline(1, color='k', ls='dashed', alpha=.7)
                
        ax_in.plot(df['Wavelength'][445:1739], norm, color='m', alpha=.7, label='Splines 1')
        ax_in.axhline(1, color='k', ls='dashed', alpha=.7)
        ax_in.set_ylim(0.8,1.05)
        ax_in.set_title('Zoom')
        ax_in.grid()
        
        ax.set_ylim(0.4,1.2)
        ax.set_ylabel('Normalized flux')
        ax.set_xlabel('Wavelength $[Å]$')

        ax.legend(loc='lower left', fontsize=8)
        ax.grid()
        
        for i, ax_ in enumerate([ax]+[ax_in]):
            ax_.grid(
            which="major",
            linewidth=np.sqrt(fig_size) * 0.015,
            color="black")
        
        plt.tight_layout()
     

     
    return norm, norm2  


#1era estrella de prueba:
#wvl_short1 = np.array([3992, 4088, 4438, 4795, 5158]) #para las estrellas de prueba
wvl_short1 = np.array([3996, 4088, 4466, 4795, 5061])
wvl_long1 = np.array([3563, 3697, 3784, 3911, 3992, 4088, 4438, 4795, 5158, 5550, 5937, 6182.8, 6444, 6961, 7380])

n1 = envelope(df1_prob, prob_name[0], wvl_short1, wvl_long1, plotting=True, resid=True, splines2=True)
    #n1[0]=norm   & n1[1]=norm2

df_norm2 = pd.DataFrame(n1[0])  
df_norm2['Wavelength'] = df1['Wavelength'][39:]

# multiple_plots(df_norm2, star_name[0], refs_dict, files)     

            
#2a estrella de prueba:      
wvl_short2 = np.array([3940, 4037, 4209, 4458, 4603, 4800, 5065])
wvl_long2 = np.array([3552, 3780, 3940, 4037, 4209, 4458, 4603, 4800, 5065, 5300, 6007, 6650, 7300])
        
n2 = envelope(df2_prob, prob_name[1], wvl_short2, wvl_long2, plotting=True, resid=True, splines2=True)
   
df_norm2_2 = pd.DataFrame(n2[0])  
df_norm2_2['Wavelength'] = df2['Wavelength'][39:]
   
# multiple_plots(df_norm2_2, star_name[1], refs_dict, files)       
        

#%%
#Código para ir comparando la estrella de prueba con cada uno de los espectros de referencia:
import shutil

def comparison(star_index, studied_norm_df, dic, path, plotting=True, index=1):
    destination = 'Desktop/Atmosferas_estelares/entregable1/espectros_referencia/candidates/'            
    new_folder = os.path.join(destination, str(index))
    os.mkdir(new_folder)

    for i in range(len(files)):
        xticks = np.arange(3900, 5100, 300) 
                
        plt.figure()
        plt.plot(studied_norm_df['Wavelength'], studied_norm_df['Flux'], color='tab:blue', label='Studied star: %s' %star_index)
        plt.plot(dic[files[i]]['Wavelength'], dic[files[i]]['Flux'], color='k', alpha=.5, label='Reference star: %s' %files[i])
        plt.axhline(1, color='k', ls='dashed', alpha=.5)
        plt.xticks(xticks)
        
        plt.xlabel('Wavelength $[Å]$')
        plt.ylabel('Flux')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        pregunta = input('Are these spectra similar? Please enter: \n' + '0 if NOT \n' +\
                         '1 if YES \n' + '\n')

        if pregunta == '0':
            print('These spectra are not similar. Keep looking for a twin jiji')
            
            
        elif pregunta == '1':
            
            origen = os.path.join(path, files[i])
            shutil.copy(origen, new_folder)
            
            print('These spectra are similar, so the reference file has been copied in CANDIDATES FOLDER.')
            
        
    n_cand, cand_dic = to_df(new_folder)
        
    if plotting == True:
        multiple_plots(studied_norm_df, star_index, cand_dic, n_cand)
            
        
    return cand_dic, n_cand



candidatos_path = 'Desktop/Atmosferas_estelares/entregable1/espectros_referencia/candidates/'

cand_title_prob = ['Candidates of starprob1', 'Candidates of starprob2']
cand_title = ['Candidates of HD78558_prueba1', 'Candidates of HD161695_prueba2']


#%% Estrellas de prueba
cand_dict1, n_cand1 = comparison(star_name[0], df_norm2, refs_dict, directory_path, index=1)
multiple_plots(df_norm2, cand_title[0], cand_dict1, n_cand1, n_col=3)


cand_dict2, n_cand2 = comparison(star_name[1], df_norm2_2, refs_dict, directory_path, index=2)
multiple_plots(df_norm2_2, cand_title[1], cand_dict2, n_cand2, n_col=4)


#%% ESTRELLAS DE ESTUDIO:

#cand_dict1, n_cand1 = comparison(prob_name[0], df_norm2, refs_dict, directory_path, index='prob1')
n_cand1, cand_dict1 = to_df('/Users/carlota/Desktop/Atmosferas_estelares/entregable1/espectros_referencia/candidates/prob1')
multiple_plots(df_norm2, cand_title_prob[0], cand_dict1, n_cand1, n_col=3)


#%% Candidatos finales para la starprob1:
n_cand3, cand_dict3 = to_df('/Users/carlota/Desktop/Atmosferas_estelares/entregable1/espectros_referencia/candidates/def_prob1')
multiple_plots(df_norm2, cand_title_prob[0], cand_dict3, n_cand3, n_col=1, spt='G', ratio='starprob1')

#%%
cand_dict2, n_cand2 = comparison(prob_name[1], df_norm2_2, refs_dict, directory_path, index='prob2')

n_cand2, cand_dict2 = to_df('/Users/carlota/Desktop/Atmosferas_estelares/entregable1/espectros_referencia/candidates/prob2')
multiple_plots(df_norm2_2, cand_title_prob[1], cand_dict2, n_cand2, n_col=3)

#%% Candidatos finales para la starprob2:
n_cand4, cand_dict4 = to_df('/Users/carlota/Desktop/Atmosferas_estelares/entregable1/espectros_referencia/candidates/def_prob2')
multiple_plots(df_norm2_2, cand_title_prob[1], cand_dict4, n_cand4, n_col=1, spt='A')



#%% COCIENTES DE INTENSIDAD DE LINEAS:
def cocientes(starprob_df, wvl_num, wvl_denom, ref=False):
    a = np.where(starprob_df['Wavelength'] >= wvl_num)
    b = np.where(starprob_df['Wavelength'] >= wvl_denom)
    
    if ref == False:
        valores = [a[0][0]+484, b[0][0]+484]

    if ref == True:
        valores = [a[0][0]+39, b[0][0]+39]
    
    return (1-starprob_df['Flux'][valores[0]]) /(1-starprob_df['Flux'][valores[1]])

    

#cociente de HeI/MgII:
a = cocientes(df_norm2_2, 4452, 4089, ref=False)
b = cocientes(cand_dict4[n_cand4[0]], 4452, 4089, ref=True)
c = cocientes(cand_dict4[n_cand4[1]], 4452, 4089, ref=True)
d = cocientes(cand_dict4[n_cand4[2]], 4452, 4089, ref=True)
e = cocientes(cand_dict4[n_cand4[3]], 4452, 4089, ref=True)


def fig_ratios(star_df, prob_name, dic, files, ratio=False):
    fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True)
    
    columns = [0,1]
    for i in columns:
        if ratio == 'starprob1':
            ax[i].plot(star_df['Wavelength'], star_df['Flux'], '-.', label='%s'%prob_name)
            ax[i].axhline(1, color='k', ls='dashed', alpha=.5)
            
            Gcommon = [4045, 4077, 4226]
            
            for y in range(len(cand_dict3)):        
                ax[i].plot(dic[files[y]]['Wavelength'], dic[files[y]]['Flux'], label='%s' %n_cand3[y])
        
            for x in range(len(Gpos)):
                if Gpos[x] in  Gcommon:                
                    ax[i].axvline(Gpos[x], color='k', ls='dashed', linewidth=1.2)
                    ax[i].axvspan(Gpos[x]-5, Gpos[x]+5, color='gray', alpha=.2)
                    ax[i].text(Gpos[x] + 0.2, 1.05, str(Getiq[x]), fontsize=10, color='k')
                else:
                    ax[i].axvline(Gpos[x], color='k', alpha=.6, ls='dashed')
                    ax[i].text(Gpos[x] + 0.1, 1.05, str(Getiq[x]), fontsize=10, color='k')
        
            ax[i].set_ylim(0.15, 1.2)
            fig.suptitle('Ratios for %s'%prob_name)
            
            if i == 0:
                ax[0].set_xlim(4045-10, 4077+10)
                ax[0].set_xlabel('Wavelength $[Å]$')
                ax[0].set_ylabel('Normalized flux')
                ax[0].legend(loc='lower center')
            else:
                ax[1].set_xlim(4226-10, 4226+10)
                ax[1].set_xlabel('Wavelength $[Å]$')
                #ax[1].legend(loc='upper right', bbox_to_anchor=(1.2, 1))


        if ratio == 'starprob2':
            ax[i].plot(star_df['Wavelength'], star_df['Flux'], '-.', label='%s'%prob_name)
            ax[i].axhline(1, color='k', ls='dashed', alpha=.5)
            
            A1 = [4089]
            A1etiq = ['SiIV']
            
            A2 = [4471,4481, 4552]
            A2etiq = ['HeI', 'MgII', 'SiIII']
            
            
            for y in range(len(cand_dict4)):        
                ax[i].plot(dic[files[y]]['Wavelength'], dic[files[y]]['Flux'], label='%s' %n_cand4[y])
                
            fig.suptitle('Ratios for %s'%prob_name)
            ax[i].set_ylim(0.5,1.2)
            
            if i == 0:
                ax[0].axvline(4089, color='k', ls='dashed', linewidth=1.2)
                ax[0].axvspan(4089-5, 4089+5, color='gray', alpha=.2)
                ax[0].text(4089 + 0.2, 1.05, 'SiIV', fontsize=10, color='k')

                ax[0].set_xlim(4089-10, 4089+10)
                ax[0].set_xlabel('Wavelength $[Å]$')
                ax[0].set_ylabel('Normalized flux')
                ax[0].legend(loc='lower center')
            else:
                for i in range(len(A2)):
                    ax[1].axvline(A2[i], color='k', ls='dashed', linewidth=1.2)
                    ax[1].axvspan(A2[i]-5, A2[i]+5, color='gray', alpha=.2)
                    ax[1].text(A2[i] + 0.2, 1.05, str(A2etiq[i]), fontsize=10, color='k')
                
                ax[1].set_xlim(4471-10, 4552+10)
                ax[1].set_xlabel('Wavelength $[Å]$')
                #ax[1].legend(loc='upper right', bbox_to_anchor=(1.2, 1))


#fig_ratios(df_norm2, prob_name[0], cand_dict3, n_cand3, ratio='starprob1')
#fig_ratios(df_norm2_2, prob_name[1], cand_dict4, n_cand4, ratio='starprob2')

#%%
fig, ax = plt.subplots(ncols=3, nrows=1, sharey=True)
ax[0].axvline(4340, color='k', ls='dashed', linewidth=1.2)
ax[0].text(4340 + 0.2, 1.02, 'H$_\\gamma$', fontsize=10, color='k')

ax[0].plot(df_norm2_2['Wavelength'], df_norm2_2['Flux'], '-.', label='starprob2')
ax[0].axhline(1, color='k', ls='dashed', alpha=.5)
ax[0].plot(cand_dict4[n_cand4[1]]['Wavelength'], cand_dict4[n_cand4[1]]['Flux'], label='HD13267_B5Ia')
ax[0].plot(cand_dict4[n_cand4[3]]['Wavelength'], cand_dict4[n_cand4[3]]['Flux'], label='HD150898_B0Ib')
ax[0].set_xlim(4340-20, 4340+20)
ax[0].set_ylabel('Normalized flux')
ax[0].set_xlabel('Wavelength $[Å]$')

ax[1].axvline(4861, color='k', ls='dashed', linewidth=1.2)
ax[1].text(4861 + 0.2, 1.02, 'H$_\\beta$', fontsize=10, color='k')

ax[1].plot(df_norm2_2['Wavelength'], df_norm2_2['Flux'], '-.', label='starprob2')
ax[1].axhline(1, color='k', ls='dashed', alpha=.5)
ax[1].plot(cand_dict4[n_cand4[1]]['Wavelength'], cand_dict4[n_cand4[1]]['Flux'], label='HD13267_B5Ia')
ax[1].plot(cand_dict4[n_cand4[3]]['Wavelength'], cand_dict4[n_cand4[3]]['Flux'], label='HD150898_B0Ib')
ax[1].legend(loc='lower center')
ax[1].set_xlim(4861-10, 4861+10)
ax[1].set_xlabel('Wavelength $[Å]$')



ax[2].axvline(4102, color='k', ls='dashed', linewidth=1.2)
ax[2].text(4102 + 0.2, 1.02, 'H$_\\delta$', fontsize=10, color='k')


ax[2].plot(df_norm2_2['Wavelength'], df_norm2_2['Flux'], '-.', label='starprob2')
ax[2].axhline(1, color='k', ls='dashed', alpha=.5)
ax[2].plot(cand_dict4[n_cand4[1]]['Wavelength'], cand_dict4[n_cand4[1]]['Flux'], label='HD13267_B5Ia')
ax[2].plot(cand_dict4[n_cand4[3]]['Wavelength'], cand_dict4[n_cand4[3]]['Flux'], label='HD150898_B0Ib')
ax[2].set_xlim(4102-10, 4102+10)
ax[2].set_xlabel('Wavelength $[Å]$')   
fig.suptitle('Balmer lines')      