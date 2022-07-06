import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import pickle
import os

workdir = os.path.dirname(os.getcwd())
srcdir = os.getcwd()
datadir = workdir + '/data/'
outputdir = workdir + '/output/'
docdir = workdir + '/doc/'

obs_series = pd.read_csv(datadir + 'data.csv', delimiter=',')
obs_series = np.array(obs_series.iloc[:,1:]).T

T = obs_series.shape[1]
N = 2000000
Λ_scale = 1.0
cd_scale = 1.0

θ_name = ['λ', 'η', \
        'b11', 'b22', \
        'As11', 'As12', 'As13',\
        'As21', 'As22', 'As23', 'Aso2', \
        'As31', 'As32', 'As33', 'Aso3', \
        'Bs11', 'Bs21', 'Bs22', 'Bs31', 'Bs32', 'Bs33',\
        'j21',  'j31',  'j32']


def return_dir(seed):
    case = 'actual data, seed = ' + str(seed) + ', T = ' + str(T) + ', N = ' + str(N) + ', Λ_scale = ' + str(Λ_scale) + ', cd_scale = ' + str(cd_scale)
    return outputdir + case  + '/'

casedir7 = return_dir(7)
casedir8 = return_dir(8)
# casedir9 = return_dir(9)

t = 100
with open(casedir7 + 'θ_'+str(t)+'.pkl', 'rb') as f:
       θ_final7 = pickle.load(f)
with open(casedir8 + 'θ_'+str(t)+'.pkl', 'rb') as f:
       θ_final8 = pickle.load(f)
# with open(casedir9 + 'θ_'+str(t)+'.pkl', 'rb') as f:
#        θ_final9 = pickle.load(f)

def return_coll(θ_final):
    λ_particle = []; η_particle = []; b11_particle = []; b22_particle = []
    As11_particle = []; As12_particle = []; As13_particle = []; 
    As21_particle = []; As22_particle = []; As23_particle = []; Aso2_particle = []
    As31_particle = []; As32_particle = []; As33_particle = []; Aso3_particle = []
    Bs11_particle = []; Bs21_particle = []; Bs22_particle = []; Bs31_particle = []; Bs32_particle = []; Bs33_particle = []
    j21_particle = []; j31_particle = []; j32_particle = [];
    λ_iter_particle = []; Ass_iter_particle = []

    for n in range(N):
        λ_particle.append(θ_final[n][1][1,1])
        η_particle.append(θ_final[n][0][1,0])
        b11_particle.append(θ_final[n][2][0,0])
        b22_particle.append(θ_final[n][2][1,1])
        As11_particle.append(θ_final[n][4][0,0])
        As12_particle.append(θ_final[n][4][0,1])
        As13_particle.append(θ_final[n][4][0,2])
        As21_particle.append(θ_final[n][4][1,0])
        As22_particle.append(θ_final[n][4][1,1])
        As23_particle.append(θ_final[n][4][1,2])
        Aso2_particle.append(θ_final[n][3][1,0])
        As31_particle.append(θ_final[n][4][2,0])
        As32_particle.append(θ_final[n][4][2,1])
        As33_particle.append(θ_final[n][4][2,2])
        Aso3_particle.append(θ_final[n][3][2,0])
        Bs11_particle.append(θ_final[n][5][0,0])
        Bs21_particle.append(θ_final[n][5][1,0])
        Bs22_particle.append(θ_final[n][5][1,1])
        Bs31_particle.append(θ_final[n][5][2,0])
        Bs32_particle.append(θ_final[n][5][2,1])
        Bs33_particle.append(θ_final[n][5][2,2])
        j21_particle.append(θ_final[n][6][1,0])
        j31_particle.append(θ_final[n][6][2,0])
        j32_particle.append(θ_final[n][6][2,1])
        λ_iter_particle.append(θ_final[n][7])
        Ass_iter_particle.append(θ_final[n][7])

    θ_coll = [λ_particle, η_particle, b11_particle, b22_particle, \
              As11_particle, As12_particle, As13_particle,\
              As21_particle, As22_particle, As23_particle, Aso2_particle,\
              As31_particle, As32_particle, As33_particle, Aso3_particle,\
              Bs11_particle, Bs21_particle, Bs22_particle, Bs31_particle, Bs32_particle, Bs33_particle,\
              j21_particle, j31_particle, j32_particle]
    return θ_coll

θ_coll7 = return_coll(θ_final7)
θ_coll8 = return_coll(θ_final8)
# θ_coll9 = return_coll(θ_final9)

# plot_series = []
# for i in tqdm(range(len(θ_name))):
#     plot_series.append(pd.DataFrame([θ_coll7[i],θ_coll8[i],θ_coll9[i]],index = ['seed = 7','seed = 8','seed = 9']).T)

plot_series = []
for i in tqdm(range(len(θ_name))):
    plot_series.append(pd.DataFrame([θ_coll7[i],θ_coll8[i]],index = ['seed = 7','seed = 8']).T)

period = t
fig, axes = plt.subplots(6,4,figsize = (15,10))
for v, ax in enumerate(axes.flatten()):
    sns.kdeplot(data = plot_series[v], ax = ax)
    ax.set_title(θ_name[v])
    if v !=0:
        ax.get_legend().remove()

title = 'Actual Data, Distribution of parameters, T = '+ str(period)+', N = '+str(N)+', Different Random seeds'
fig.suptitle(title, fontsize=16)    
fig.tight_layout()
fig.savefig(docdir + title + ', Λ_scale = ' + str(Λ_scale) + ', cd_scale = ' + str(cd_scale) + '.png',dpi = 400, bbox_inches = "tight")