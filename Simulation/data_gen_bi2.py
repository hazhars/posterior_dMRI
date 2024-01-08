import numpy as np
import torch
import os
import sys
import pickle
sys.path.append(os.path.abspath('/BayesFlow'))
from sklearn.mixture import GaussianMixture
import time
import matplotlib.pyplot as plt
# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi import inference
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

torch.manual_seed(0)

b_vals=np.load('101006_bval.npy')
b_vecs=np.load('101006_bvec.npy')
ng=108

from dipy.core.gradients import gradient_table
from dipy.reconst.shm import QballModel

from functions import find_b0, normalize_data, coef_calc, ODF_calc, orth_vectors, eigs2tensor, spher2cart, FA_calc, FA_calc_coef , ang_b


#######################################################################################################################
t1,t2=orth_vectors(b_vecs[:,0])
min_ang=0
max_ang=np.pi
min_val=1
max_val=2.5
min_val3=0.1
max_val3=0.6
eig2=1
exp_len=500

prior_min = [min_ang, min_ang]
prior_max = [max_ang, max_ang]
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                    high=torch.as_tensor(prior_max))

an=np.linspace(0,np.pi, num=32,endpoint=True)
lnan=len(an)
pr=np.append(np.zeros(lnan).reshape(lnan,1),an.reshape(lnan,1),axis=1)

for i in range(1,len(an)):
    pr=np.append(pr,np.append(an[i]*np.ones(lnan).reshape(lnan,1),an.reshape(lnan,1),axis=1),axis=0)
pr=pr.astype(np.float32)


######################################

def simulator(parameter_set,bvl,bvc,ng,nc=0):
    p1 = np.asarray(parameter_set)
    p = np.ones(14)
    p[3:7] = p1[2:6]
    p[10:14] = p1[8:12]
    p[0:3] = spher2cart(p1[0:2])
    p[7:10] = spher2cart(p1[6:8])

    #creating tensors
    D1=eigs2tensor(p[0:5])
    D1_s = eigs2tensor(np.concatenate((p[0:3],p[5:7])))
    D2=eigs2tensor(p[7:12])
    D2_s = eigs2tensor(np.concatenate((p[7:10], p[12:14])))


    s1= np.ones(ng)
    s2= np.ones(ng)
    s_o=np.ones(90)
    k=0

    for i in range(0,ng):
        s1[i]=0.7*np.exp(-bvl[0,i]*np.dot(np.dot(bvc[:, i].reshape(1,3),D1),bvc[:, i].reshape(3,1)))+0.3*np.exp(-bvl[0,i]*np.dot(np.dot(bvc[:, i].reshape(1,3),D1_s),bvc[:, i].reshape(3,1)))
        s2[i]=0.7*np.exp(-bvl[0,i]*np.dot(np.dot(bvc[:, i].reshape(1,3),D2),bvc[:, i].reshape(3,1)))+0.3*np.exp(-bvl[0,i]*np.dot(np.dot(bvc[:, i].reshape(1,3),D2_s),bvc[:, i].reshape(3,1)))

        if bvl[0, i] > 10:
            noise_term=nc * np.random.randn(1)
            s1[i] = s1[i] + noise_term
            s2[i] = s2[i] + noise_term
            s_o[k]=0.5 * s1[i] + 0.5 * s2[i]
            k=k+1


    return 0.5*s1+0.5*s2,s_o


def test_prior(batch_size):
    p_samples = np.random.uniform(low=(min_ang, min_ang, min_val, 0.1, min_val3, 0.1, min_ang, min_ang, min_val, 0.1, min_val3, 0.1),
                                  high=(max_ang, max_ang, max_val, eig2, max_val3, 0.8, max_ang, max_ang, max_val,eig2, max_val3, 0.8), size=(batch_size, 12))
    return p_samples.astype(np.float32)
# test_vec=test_prior(1).reshape(100,)
#
# test_vec=[np.pi/4,np.pi/4,2.7,1.5,0.6,0.1,3*np.pi/4,np.pi/4,2./7,1.5,0.6,0.1]
# angle=(180/np.pi)*ang_b(test_vec[0:2],test_vec[4:6])




#################################################################################
import dipy.reconst.dti as dti

gtab = gradient_table(b_vals.reshape(108), b_vecs.T)
tenmodel = dti.TensorModel(gtab)

theta = np.repeat(an,32)
phi = np.tile(an,32)
hsph_i = Sphere(theta=theta, phi=phi)

###############################################################################################################
Y_odf=np.load('Y_odf_1024_45.npy')
P=np.array([ 1.   , -0.5      , -0.5     , -0.5    , -0.5      ,
       -0.5      ,  0.375    ,  0.375    ,  0.375    ,  0.375    ,
        0.375    ,  0.375    ,  0.375    ,  0.375    ,  0.375    ,
       -0.3125   , -0.3125   , -0.3125   , -0.3125   , -0.3125   ,
       -0.3125   , -0.3125   , -0.3125   , -0.3125   , -0.3125   ,
       -0.3125   , -0.3125   , -0.3125   ,  0.2734375,  0.2734375,
        0.2734375,  0.2734375,  0.2734375,  0.2734375,  0.2734375,
        0.2734375,  0.2734375,  0.2734375,  0.2734375,  0.2734375,
        0.2734375,  0.2734375,  0.2734375,  0.2734375,  0.2734375])
#####################################################################################################################
an=np.linspace(0,np.pi, num=32,endpoint=True)
lnan=len(an)
pr=np.append(np.zeros(lnan).reshape(lnan,1),an.reshape(lnan,1),axis=1)

for i in range(1,len(an)):
    pr=np.append(pr,np.append(an[i]*np.ones(lnan).reshape(lnan,1),an.reshape(lnan,1),axis=1),axis=0)
pr=pr.astype(np.float32)

bv_new=np.ones([3,len(pr)])
for i in range(0,len(pr)):
    bv_new[:,i]=spher2cart(pr[i,:])
#
bvals_new=3000*np.ones([1,len(pr)])
ng=108
noise_c=0.0

test_vec=test_prior(5000000)
theta=np.ones([len(test_vec),12])
j=0
for k in range(len(test_vec)):
    if ang_b(test_vec[k,0:2],test_vec[k,6:8])>np.pi/18  and FA_calc_coef(test_vec[k,2:4])>0.2 and FA_calc_coef(test_vec[k,8:10])>0.2 and test_vec[k,4]<0.5*test_vec[k,2] and test_vec[k,10]<0.5*test_vec[k,8]:
        theta[j, :] = test_vec[k, :]
        j+=1
theta=theta[0:j,:]
horiz=len(theta)
# horiz=300
SH=np.ones([horiz,3,45])
sig,sig1= simulator(theta[0, :], (1/3)*b_vals, b_vecs, ng, noise_c)
_,SH_45_mat=coef_calc((1/3)*b_vals.reshape((ng,)), np.transpose(b_vecs), sig.reshape((1, ng)))

start_time = time.time()


for i in range(0,horiz):
    _,sig1= simulator(theta[i, :], (1/3)*b_vals, b_vecs, ng, 0.053/3)
    _,sig2= simulator(theta[i, :], (2/3)*b_vals, b_vecs, ng, 0.053/3)
    _,sig3= simulator(theta[i, :], b_vals, b_vecs, ng, 0.053/3)
    SH[i,0,:]= np.dot(np.concatenate((sig1,sig1)), SH_45_mat.T)
    SH[i,1,:]= np.dot(np.concatenate((sig2,sig2)), SH_45_mat.T)
    SH[i,2,:]= np.dot(np.concatenate((sig3,sig3)), SH_45_mat.T)
    # S1[i,:]=sig1
    # S2[i, :] = sig2
    # S3[i, :] = sig3
    # odf1 =ODF_calc(A,P,Y_odf)
    # odf[i,:,:]=odf1.reshape(32,32)

g=time.time()
g1=g- start_time
print("simulation/training time: %s minutes ---" % (g1/60))
# plt.imshow(odf[0,:,:])
# plt.show()

torch.save(torch.from_numpy(SH.astype(np.float32)),  'SH_SNR3_bi_3.pt')
torch.save(torch.from_numpy(theta.astype(np.float32)),  'theta_SNR3_bi_3.pt')


# plt.figure(1)
# plt.imshow(B.reshape([32,32]),cmap='jet')
# plt.show()
# plt.colorbar()
#
# # odf1 = 0.5 * tenmodel.fit(s1).odf(hsph_i) + 0.5 * tenmodel.fit(s2).odf(hsph_i)
# plt.figure(2)
# plt.imshow(x[i,:,:],cmap='jet')
# plt.show()
# plt.colorbar()

