import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import QballModel
coef=1000


def find_b0(dwi, where_b0, mask= None):
    b0= dwi[...,where_b0].mean(-1)
    np.nan_to_num(b0).clip(min= 0., out= b0)

    if mask is not None:
        return b0*mask
    else:
        return b0


# Normalization Function
def normalize_data(dwi, where_b0= None, mask= None, b0= None):

    dwi= dwi.astype('float32')

    if where_b0 is not None and b0 is None:
        b0= find_b0(dwi, where_b0, mask)
        np.nan_to_num(b0).clip(min=1., out=b0) # can be changed to 0. as well
        for i in where_b0:
            dwi[...,i] = b0
    else:
        np.nan_to_num(b0).clip(min=1., out=b0) # can be changed to 0. as well


    # when b0 is clipped to 1.
    dwiPrime= dwi/b0[...,None]
    # when b0 is clipped to 0.
    # dwiPrime= dwi/b0[...,None].clip(min=1.)
    np.nan_to_num(dwiPrime).clip(min=0., max=1., out= dwiPrime)

    # if mask is not None:
    #     dwiPrime= applymask(dwiPrime, mask)

    return (dwiPrime, b0)



def coef_calc(bvals,bvecs,data,N_shm=8):
    bvals = np.append(bvals, bvals)
    bvecs = np.append(bvecs, -bvecs, axis=0)
    data = np.append(data, data, axis=1)

    # gtab = gradient_table(bvals, bvecs, b0_threshold=B0_THRESH)
    gtab = gradient_table(bvals, bvecs)
    qb_model = QballModel(gtab, sh_order=N_shm)

    # b0 = find_b0(data, where_b0=np.where(qb_model.gtab.b0s_mask)[0])

    # inserting correct shm_coeff computation block ---------------------------------
    smooth = 0.006
    # data = applymask(data, mask_data)
    data_norm, _ = normalize_data(data, where_b0=np.where(qb_model.gtab.b0s_mask)[0])

    L = qb_model.n * (qb_model.n + 1)
    L **= 2
    _fit_matrix = np.linalg.pinv(qb_model.B.T @ qb_model.B + np.diag(smooth * L)) @ qb_model.B.T
    shm_coeff = np.dot(data_norm[..., qb_model._where_dwi], _fit_matrix.T)
    # shm_coeff = applymask(shm_coeff, mask_data)

    # P=np.ones(len(L))
    # P[1:len(L)]=qb_model.n[1:len(L)]
    # for k in range(1,len(L)):
    #     P[k]=((-1)**(P[k]/2))*(np.prod(np.arange(1,P[k],2)))/(np.prod(np.arange(2,P[k]+1,2)))

    # CW=2*np.pi*P*shm_coeff[0,:]
    # ODF=np.matmul(qb_model.B, CW)
    return shm_coeff ,_fit_matrix

def ODF_calc(shm_coeff,P,Y_odf):
    CW = 2 * np.pi * P * shm_coeff[0, :]
    ODF = np.matmul(Y_odf, CW)
    return ODF



def orth_vectors(v):
    z = np.random.randn(3)  # take a random vector
    z -= z.dot(v) * v  # make it orthogonal to k
    z /= np.linalg.norm(z)
    y = np.cross(v, z)
    y/= np.linalg.norm(y)
    return z,y


def eigs2tensor(p):
    v11=p[0:3]
    v11/=np.linalg.norm(v11)
    v12,v13=orth_vectors(v11)
    S1=np.append(v11.reshape(3,1), v12.reshape(3,1), axis=1)
    S1 = np.append(S1, v13.reshape(3,1), axis=1)
    M1=1/coef*np.diag([p[3],p[4]*p[3],p[4]*p[3]])
    D1=np.dot(np.dot(S1,M1),np.transpose(S1))

    return D1

def spher2cart(ang):
    # Here ang[0] is the angle with z axis, (theta in physics)
    cart=np.ones(3)
    cart[0]=np.sin(ang[0])*np.cos(ang[1])
    cart[1] = np.sin(ang[0]) * np.sin(ang[1])
    cart[2] = np.cos(ang[0])
    return cart

def FA_calc(eig):
    FA=np.sqrt(((eig[1]-eig[0])**2)/(2*eig[1]**2+eig[0]**2))
    return FA

def FA_calc_coef(eig):
    FA=np.sqrt(((eig[1]*eig[0]-eig[0])**2)/(2*(eig[0]*eig[1])**2+eig[0]**2))
    return FA

def ang_b(orig,result):
    # tes=tes.reshape(8)
    A1 = spher2cart(orig)
    A2 = spher2cart(result)
    ang_betw = np.arccos(np.dot(A1, A2))
    if ang_betw>np.pi/2:
        ang_betw=ang_betw-2*(ang_betw-np.pi/2)
    return ang_betw