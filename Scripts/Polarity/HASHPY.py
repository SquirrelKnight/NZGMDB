import pandas as pd
import numpy as np
import numba
from math import sqrt, cos, sin, atan, atan2, acos, asin, pi

class TooManyRots(Exception):
    pass

def fpcoor_2SDR(fnorm, slip):
    degrad = 180/np.pi
    if 1 - abs(fnorm[2]) < 1e-7:
        print('***FPCOOR warning, horz fault, strike undefined')
        delt = 0
        phi = atan2(-slip[0],slip[1])
        clam = cos(phi) * slip[0] + sin(phi) * slip[1]
        slam = sin(phi) * slip[0] - cos(phi) * slip[1]
        lam = atan2(slam, clam)
    else:
        phi = atan2(-fnorm[0], fnorm[1])
        a = sqrt(fnorm[0] ** 2 + fnorm[1] ** 2)
        delt = atan2(a, -fnorm[2])
        clam = cos(phi) * slip[0] + sin(phi) * slip[1]
        slam = -slip[2] / sin(delt)
        lam = atan2(slam, clam)
        if delt > 0.5 * pi:
            delt = pi - delt
            phi = phi + pi
            lam = -lam
    strike = phi * degrad
    if strike < 0:
        strike = strike + 360
    dip = delt * degrad
    rake = lam * degrad
    if rake <= -180:
        rake = rake + 360
    if rake > 180:
        rake = rake - 360
    return strike,dip,rake

def fpcoor_2FS(strike, dip, rake):
    degrad = 180/pi
    phi = strike/degrad
    delt = dip/degrad
    lam = rake/degrad

    fnorm = -sin(delt) * sin(phi), sin(delt) * cos(phi), -cos(delt)
    slip = cos(lam) * cos(phi) + cos(delt) * sin(lam) * sin(phi), \
        cos(lam) * sin(phi) - cos(delt) * sin(lam) * cos(phi), \
        -sin(lam) * sin(delt)
        
    return fnorm, slip           

def cross(v1,v2):
    v3 = [v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]]  
    return v3

def to_car(the,phi,r):
    rad = np.pi/180
    z = -r * cos(the * rad)
    x = r * sin(the * rad) * cos(phi*rad)
    y = r * sin(the * rad) * sin(phi*rad)
    return x,y,z

def focal_arrays(ncoor, dang):
#     dangold = -1
    ntab = 180
    degrad = 180/np.pi
    rad = 1/degrad
    # Coordinate transform arrays
    b1 = np.empty([3, ncoor])
    b2 = np.empty([3, ncoor])
    b3 = np.empty([3, ncoor])
    # P and S amplitude arrays
    thetable = np.empty(2*ntab+1)
    phitable = np.empty([2*ntab+1,2*ntab+1])
    amptable = np.empty([2,ntab+1,2*ntab+1])
    
#     if dangold != dang:
    irot = 0
    # Set up array with direction cosines for all coordinate transformations
    for ithe in range(0,int(90.1/dang) + 1):
        the = ithe*dang
        rthe = the/degrad
        costhe = cos(rthe)
        sinthe = sin(rthe)
        fnumang = 360/dang
        numphi = round(fnumang*sinthe)
#                 numphi = fnumang
        if numphi != 0:
            dphi = 360/numphi
        else:
            dphi = 10000
        for iphi in range(0,int(359.9/dphi) + 1):
            phi = iphi*dphi
            rphi = phi/degrad
            cosphi = cos(rphi)
            sinphi = sin(rphi)
            bb3 = [sinthe*cosphi,sinthe*sinphi,costhe]
            bb1 = [costhe*cosphi,costhe*sinphi,-sinthe]
            bb2 = cross(bb3,bb1)
            for izeta in range(0,int(179.9/dang) + 1):
                zeta = izeta*dang
                rzeta = zeta/degrad
                coszeta = cos(rzeta)
                sinzeta = sin(rzeta)
                if irot > ncoor:
                    raise TooManyRots
                b3[2, irot] = bb3[2]
                b3[0, irot] = bb3[0]
                b3[1, irot] = bb3[1]
                b1[0, irot] = bb1[0] * coszeta + bb2[0] * sinzeta
                b1[1, irot] = bb1[1] * coszeta + bb2[1] * sinzeta                
                b1[2, irot] = bb1[2] * coszeta + bb2[2] * sinzeta
                b2[0, irot] = bb2[0] * coszeta - bb1[0] * sinzeta
                b2[1, irot] = bb2[1] * coszeta - bb1[1] * sinzeta                
                b2[2, irot] = bb2[2] * coszeta - bb1[2] * sinzeta
                irot = irot+1
    nrot = irot
#     dangold=dang

    astep=1/ntab
    for i in range(0,2*ntab+1):
        bbb3 = -1 + (i) * astep
        thetable[i] = acos(bbb3)
        for j in range(0,2*ntab+1):
            bbb1 = -1 + (j) * astep
            phitable[i,j] = atan2(bbb3,bbb1)
            if phitable[i,j] < 0:
                phitable[i,j] = phitable[i,j] + 2 * pi

    for i in range(0, 2*ntab+1):
        phi = (i) * np.pi * astep
        for j in range(0,ntab+1):
            theta = (j) * pi * astep
            amptable[0,j,i] = abs(sin(2 * theta) * cos(phi))
            s1 = cos(2 * theta) * cos(phi)
            s2 = -cos(theta) * sin(phi)
            amptable[1,j,i] = sqrt(s1 ** 2 + s2 ** 2)
    
    return nrot, thetable, phitable, amptable, b1, b2, b3, astep, ntab


def focal_amp_mc(ncoor, nrot, thetable, phitable, amptable, p_azi_mc, p_the_mc, sp_amp, p_pol, 
    npsta, nmc, nextra, ntotal, qextra, qtotal, maxout, arr_data, b1, b2, b3, astep, ntab):
    #     ncoor = 729000

    # Misfit arrays
#     qmis = np.empty(ncoor)
#     nmis = np.empty(ncoor)
    irotgood = np.empty(ncoor,dtype='int')
    irotgood2 = np.empty(ncoor,dtype='int')
        
    for irot in range(0,nrot): # IS THIS NECESSARY?
        irotgood[irot] = 0

    try:        

        # loop over multiple trials, this is the slowest part of the process
        for im in range(0,nmc):
            nmis0min=1.0e5
            qmis0min=1.0e5
            qmis = np.zeros(nrot)
            nmis = np.zeros(nrot)
            p_a1 = np.empty(npsta)
            p_a2 = np.empty(npsta)
            p_a3 = np.empty(npsta)
            # Convert data to Cartesian coordinates
            for idx in range(len(p_azi_mc)):
                p_a1[idx], p_a2[idx], p_a3[idx] = to_car(p_the_mc[idx][im], p_azi_mc[idx][im], 1)
            # Find misfit for each solution and minimum misfit
            p_a = np.array([p_a1,p_a2,p_a3])
            p_b1 = p_a.T @ b1
            p_b3 = p_a.T @ b3
            
            # Run computations for sp_amps
            spr_mask = (sp_amp != 0) & (~np.isnan(sp_amp))
            spr_count = sum(spr_mask)
            p_proj1 = p_a1[spr_mask].reshape(spr_count,1) - p_b3[spr_mask] * b3[0, :]
            p_proj2 = p_a2[spr_mask].reshape(spr_count,1) - p_b3[spr_mask] * b3[1, :]
            p_proj3 = p_a3[spr_mask].reshape(spr_count,1) - p_b3[spr_mask] * b3[2, :]
            plen = np.sqrt(p_proj1 ** 2 + p_proj2 ** 2 + p_proj3 ** 2)
            p_proj1 = p_proj1/plen
            p_proj2 = p_proj2/plen
            p_proj3 = p_proj3/plen
            pp_b1 = b1[0, :] * p_proj1 + b1[1,:] * p_proj2 + b1[2, :] * p_proj3
            pp_b2 = b2[0, :] * p_proj1 + b2[1,:] * p_proj2 + b2[2, :] * p_proj3
            i = np.round((p_b3[spr_mask]+1)/astep).astype('int')
            theta = thetable[i]             
            i = np.round((pp_b2+1)/astep).astype('int')
            j = np.round((pp_b1+1)/astep).astype('int')
            phi = phitable[i,j]
            i = np.round(phi/(np.pi*astep)).astype('int')
            i[i > 2 * ntab] = 0
            j = np.round(theta/(np.pi*astep)).astype('int')
            j[j > ntab] = 0
            p_amp = amptable[0,j,i]
            s_amp = amptable[1,j,i]
            sp_ratio = np.zeros(p_amp.shape)
            sp_ratio_mask = (p_amp != 0) & (s_amp != 0)
            sp_ratio[sp_ratio_mask] = np.log10(4.9 * s_amp[sp_ratio_mask] / p_amp[sp_ratio_mask])
            sp_ratio[p_amp == 0] = 4
            sp_ratio[s_amp == 0] = -2
            qmis = np.nansum(abs(sp_amp[spr_mask].reshape(spr_count,1) - sp_ratio),axis=0)
            
            # Run computations for p polarities
            p_pol_mask = p_pol != 0
            prod = p_b1[p_pol_mask] * p_b3[p_pol_mask]
            ipol = np.full(prod.shape,-1)
            ipol[prod > 0] = 1
            p_pol_compare = p_pol[p_pol_mask].reshape(len(ipol),1)
            nmis = np.sum(ipol != p_pol_compare,axis=0)
            
            if nmis.min() < nmis0min:
                nmis0min = nmis.min()
            if qmis.min() < qmis0min:
                qmis0min = qmis.min()
        
            nmis0max = ntotal
            if nmis0max < nmis0min + nextra:
                nmis0max = nmis0min + nextra
            qmis0max = qtotal
            if qmis0max < qmis0min + qextra:
                qmis0max = qmis0min + qextra
        
            # Find rotations meeting criteria
            irotgood[(nmis <= nmis0max) & (qmis < qmis0max)] = 1
            nadd = len(irotgood[irotgood == 1])
        
            # If none meet criteria
            if nadd == 0:
                qmis0min=1.0e5
                for irot in range(0,nrot):
                    if (nmis[irot] <= nmis0max) and (qmis[irot] < qmis0min):
                        qmis0min = qmis[irot]
                qmis0max = qtotal
                if qmis0max < qmis0min + qextra:
                    qmis0max = qmis0min + qextra
                
                irotgood[(nmis <= nmis0max) & (qmis < qmis0max)] = 1
                nadd = len(irotgood[irotgood == 1])
   
        nfault = irotgood[irotgood > 0].sum()
        print('...Total number of accepted fault solutions: '+str(nfault))
        irotgood2 = np.where(irotgood > 0)[0]
    
        nf = 0
        faults = []
        slips = []
        strike = []
        dip = []
        rake = []
    
        if nfault <= maxout:
            for i in range(0, nfault):
                irot = irotgood2[i]
                nf = nf + 1
                faultnorm = b3[0, irot], b3[1, irot], b3[2, irot]
                slip = b1[0, irot], b1[1, irot], b1[2, irot]
    #                 for m in range(0,3):
                faults.append(faultnorm)     
                slips.append(slip)
                s1, d1, r1 = fpcoor_2SDR(faultnorm, slip)
                strike.append(s1)
                dip.append(d1)
                rake.append(r1)
        else:
            for i in range(0, 99999):
                fran = np.random.rand(1)
                iscr = int(fran * nfault - 0.5)
                if iscr < 1:
                    iscr = 1
                if iscr > nfault:
                    iscr = nfault
                if irotgood2[iscr] <= 0:
                    continue
                irot = irotgood2[iscr]
                irotgood2[iscr] = -1
                nf = nf + 1
                faultnorm = b3[0, irot], b3[1, irot], b3[2, irot]
                slip = b1[0, irot], b1[1, irot], b1[2, irot]
                faults.append(faultnorm)     
                slips.append(slip)                
                s1, d1, r1 = fpcoor_2SDR(faultnorm, slip)
                strike.append(s1)
                dip.append(d1)
                rake.append(r1)
                if nf == maxout:
                    break
        return nf, np.array(strike), np.array(dip), np.array(rake), np.array(faults), np.array(slips)
    except TooManyRots:
        print('...focal error: # of rotations too big')

def mech_rot_2(in_data):
    norm1 = in_data[:,0:3]
    norm2 = in_data[:,3:6]
    slip1 = in_data[:,6:9]
    slip2 = in_data[:,9:12]
    
    degrad = 180/np.pi
    rotemp = np.zeros((len(norm1),4))
    for itr in range(0,4):
        n1 = np.empty((len(norm1),3))
        n2 = np.empty((len(norm1),3))
        if itr < 2:
            norm2_temp = norm2
            slip2_temp = slip2
        else:
            norm2_temp = slip2
            slip2_temp = norm2
        if (itr == 1) or (itr == 3):
            norm2_temp = -norm2_temp
            slip2_temp = -slip2_temp
        
        B1 = np.cross(norm1, slip1)
        B2 = np.cross(norm2_temp, slip2_temp)
        
        phi1 = np.dot(norm1,norm2_temp.T)[0]
        phi2 = np.dot(slip1,slip2_temp.T)[0]
        phi3 = np.dot(B1,B2.T)[0]
        
        # In some cases, identical dot products produce values incrementally higher than 1
        phi1[phi1 > 1] = 1
        phi2[phi2 > 1] = 1
        phi3[phi3 > 1] = 1
        phi1[phi1 < -1] = -1
        phi2[phi2 < -1] = -1
        phi3[phi3 < -1] = -1
        
        phi = np.array((np.arccos(phi1), np.arccos(phi2), np.arccos(phi3)))
                
        ##### !!!! Need to and an if statement to work with the else
        # Find difference vectors - the rotation axis must be orthogonal to all three of
        # these vectors
#         else:
        n = np.array([np.array(norm1) - np.array(norm2_temp), np.array(slip1) - np.array(slip2_temp), np.array(B1) - np.array(B2)])
#         n = np.array([np.array(norm1[0]) - np.array(norm2_temp[0]), np.array(slip1[0]) - np.array(slip2_temp[0]), np.array(B1[0]) - np.array(B2[0])])
        scale = np.sqrt(np.sum(n ** 2,axis = 0))
        n = n/scale
        qdot = np.array((n[0,:,1] * n[0,:,2] + n[1,:,1] * n[1,:,2] + n[2,:,1] * n[2,:,2], \
            n[0,:,0] * n[0,:,2] + n[1,:,0] * n[1,:,2] + n[2,:,0] * n[2,:,2], \
            n[0,:,0] * n[0,:,1] + n[1,:,0] * n[1,:,1] + n[2,:,0] * n[2,:,1]))
#         qdot = np.array((n[0,1] * n[0,2] + n[1,1] * n[1,2] + n[2,1] * n[2,2], \
#             n[0,0] * n[0,2] + n[1,0] * n[1,2] + n[2,0] * n[2,2], \
#             n[0,0] * n[0,1] + n[1,0] * n[1,1] + n[2,0] * n[2,1]))
                    
        # Use the two largest difference vectors, as long as they aren't orthogonal
        iout = np.argmin(scale,axis=1)
#             iout = -1
#             iout = np.full(len(norm1),-1)
        for i in range(0,3):
            iout[qdot[i] > 0.9999] = i
        n1 = np.zeros((n.shape[1],3))
        n2 = np.zeros((n.shape[1],3))
        for l in range(n.shape[1]):
            k = 1
            for j in range(0,3):
                if j != iout[l]:
                    if k == 1:
                        n1[l] = n[:,l,j]
                        k = 2
                    else:
                        n2[l] = n[:,l,j]
        # Find rotation axis by taking cross product
        R = np.cross(n1,n2)
#             scaleR = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
        scaleR = np.sqrt(np.sum(R ** 2,axis = 1))
        R = (R.T / scaleR).T
        
        # Find rotation using axis furthest from rotation axis
        theta = np.array([np.arccos(np.dot(norm1,R.T)[0]), \
            np.arccos(np.dot(slip1,R.T)[0]), \
            np.arccos(np.dot(B1,R.T)[0])])
#             qmindif = 1000
#             for i in range(0,3):
#                 if abs(theta[i] - np.pi/2) < qmindif:
#                     qmindif = abs(theta[i] - np.pi/2)
#                     iuse = i
        qmindifs = abs(theta - np.pi/2)
#         qmindif = qmindifs.max() # Changed this to find furthest from rot axis (max instead of min)
#         iuse = np.where(np.isin(qmindifs,np.max(qmindifs,axis = 0))) # or max!?
        iuse = np.argmin(qmindifs[0:2],axis=0) # Pick the minimum from either the norm or slip axes
        
#         (np.sqrt(np.sum(B1 ** 2,axis=1)) * np.sqrt(np.sum(R ** 2, axis=1)))

        rotemp[:,itr] = (np.cos(phi[iuse,np.arange(len(iuse))]) - np.cos(theta[iuse,np.arange(len(iuse))]) * np.cos(theta[iuse,np.arange(len(iuse))])) \
            / (np.sin(theta[iuse,np.arange(len(iuse))]) ** 2)
#         rotemp[:,itr] = (np.cos(phi[1,:]) - np.cos(theta[iuse,np.arange(len(iuse))]) * np.cos(theta[iuse,np.arange(len(iuse))])) \
#             / (np.sin(theta[iuse,np.arange(len(iuse))]) ** 2)

#         rotemp[:,itr] = (np.cos(phi[0,:]) - np.cos(theta[iuse,:]) * np.cos(theta[iuse,:])) \
#             / (np.sin(theta[0,:]) ** 2)
        rotemp[rotemp[:,itr] > 1,itr] = 1
        rotemp[rotemp[:,itr] < -1,itr] = -1
        rotemp[:,itr] = degrad * np.arccos(rotemp[:,itr])
        
        # If the mechanisms are very close, rotation = 0
        phi_mask = np.logical_and(np.logical_and(phi[0] < 1e-4, phi[1] < 1e-4), phi[2] < 1e-4)
        rotemp[phi_mask,itr] = 0

        # If one vector is the same, it is the rotation axis
        phi_mask = (phi[0] < 1e-4) & (phi[1] >= 1e-4) & (phi[2] >= 1e-4)
        rotemp[phi_mask,itr] = degrad * phi[1,phi_mask]
        phi_mask = (phi[0] >= 1e-4) & (phi[1] < 1e-4) & (phi[2] >= 1e-4)
        rotemp[phi_mask,itr] = degrad * phi[2,phi_mask]
        phi_mask = (phi[0] >= 1e-4) & (phi[1] >= 1e-4) & (phi[2] < 1e-4)
        rotemp[phi_mask,itr] = degrad * phi[0,phi_mask]
        
    # Find the minimum rotation for the 4 combos and change norm2 and slip2
    rota = 180
    irot = np.argmin(rotemp,axis=1)
    rota = rotemp[np.arange(len(irot)),irot]
    norm2_out = np.zeros(norm2.shape)
    slip2_out = np.zeros(slip2.shape)
    norm2_out[irot < 2] = norm2[irot < 2]
    slip2_out[irot < 2] = slip2[irot < 2]
    norm2_out[irot >= 2] = slip2[irot >= 2]
    slip2_out[irot >= 2] = norm2[irot >= 2]
    norm2_out[(irot == 1) | (irot == 3)] = -norm2_out[(irot == 1) | (irot == 3)]
    slip2_out[(irot == 1) | (irot == 3)] = -slip2_out[(irot == 1) | (irot == 3)]

    return rota, norm2_out, slip2_out

def mech_avg(nc,norm1,norm2):
    degrad = 180/np.pi
    
    # If there is only one mechanism, return that mechanism
    if nc <= 1:
        norm1_avg = norm1[0]
        norm2_avg = norm2[0]
    else:
        norm1_avg = norm1[0].copy()
        norm2_avg = norm2[0].copy()
        ref1 = norm1[0]
        ref2 = norm2[0]
        
        temp1 = norm1[1:nc]
        temp2 = norm2[1:nc]

        test = np.hstack([np.broadcast_to(ref1,(nc-1,3)),temp1,np.broadcast_to(ref2,(nc-1,3)),temp2])
        rota,temp1,temp2 = mech_rot_2(test)
        norm1_avg = norm1_avg + np.sum(temp1,axis = 0)
        norm2_avg = norm2_avg + np.sum(temp2,axis = 0)
        ln_norm1 = np.sqrt(np.sum(norm1_avg ** 2))
        ln_norm2 = np.sqrt(np.sum(norm2_avg ** 2))
        norm1_avg = norm1_avg / ln_norm1
        norm2_avg = norm2_avg / ln_norm2
       
        avang1 = 0
        avang2 = 0
        temp1 = norm1[0:nc]
        temp2 = norm2[0:nc]

        test = np.hstack([np.broadcast_to(norm1_avg,(nc,3)),temp1,np.broadcast_to(norm2_avg,(nc,3)),temp2])
        rota,temp1,temp2 = mech_rot_2(test)
        d11 = np.dot(temp1,norm1_avg)
        d22 = np.dot(temp2,norm2_avg)
        d11[d11 >= 1] = 1
        d11[d11 <= -1] = -1
        d22[d22 >= 1] = 1
        d22[d22 <= -1] = -1
        a11 = np.arccos(d11)
        a22 = np.arccos(d22)
        avang1 = np.sum(a11 ** 2)
        avang2 = np.sum(a22 ** 2)
        avang1 = np.sqrt(avang1/nc)
        avang2 = np.sqrt(avang2/nc)

        
        # The average normal vectors may not be exactly orthogonal (although they are
        # usually very close) - find the misfit from orthogonal and adjust the vectors to
        # to make them orthogonal - adjust the more poorly constrained plane more
        
        if avang1 + avang2 >= 0.0001:
            maxmisf = 0.01
            fract1 = avang1/(avang1 + avang2)
            for icount in range(0,100):
                dot1 = np.dot(norm1_avg,norm2_avg)
                misf = 90 - acos(dot1) * degrad
                if abs(misf) <= maxmisf:
                    break
                theta1 = misf * fract1 / degrad
                theta2 = misf * (1 - fract1) / degrad
                temp = norm1_avg
                norm1_avg = norm1_avg - norm2_avg * sin(theta1)
                norm2_avg = norm2_avg - temp * sin(theta2)
                ln_norm1 = sqrt(np.sum(norm1_avg ** 2))
                ln_norm2 = sqrt(np.sum(norm2_avg ** 2))
                norm1_avg = norm1_avg / ln_norm1
                norm2_avg = norm2_avg / ln_norm2
    return np.array(norm1_avg), np.array(norm2_avg)

def mech_prob(cangle,prob_max,nf,norm1in,norm2in):
    degrad = 180 / np.pi
    rms_diff = np.zeros([2,5])
    
    if nf <= 1: ##### !!!! This may need to be fixed
        norm1_avg = norm1in[0]
        norm2_avg = norm2in[0]
        str_avg, dip_avg, rak_avg = fpcoor_2SDR(norm1_avg, norm2_avg)
        prob = 1
        rms_diff[0] = 0
        rms_diff[1] = 0
        nsltn = 1
#         return nsltn,str_avg,dip_avg,rak_avg,prob,rms_diff
    else:
        norm1 = norm1in.copy()
        norm2 = norm2in.copy()
        nfault = nf
        nc = nf
        prob = np.zeros(5)
        str_avg = np.zeros(5)
        dip_avg = np.zeros(5)
        rak_avg = np.zeros(5)
        for imult in range(0,5):
            if nc < 1: # ??? not sure about this if statement
                nsltn = imult - 1 # Use only ones with certain probability
                break
            for icount in range(0,nf):
                print(icount)
#                 rota = np.empty(nc)
                norm1_avg, norm2_avg = mech_avg(nc, norm1, norm2)
#                 print(fpcoor_2SDR(norm1_avg,norm2_avg))
                temp1 = norm1
                temp2 = norm2

                test = np.hstack([np.broadcast_to(norm1_avg,(nc,3)),temp1[0:nc],np.broadcast_to(norm2_avg,(nc,3)),temp2[0:nc]])
                rota = mech_rot_2(test)[0]
                maxrot = abs(rota[0:nc]).max()
                imax = np.argmax(rota)
                if maxrot <= cangle:
                    break
                nc = nc - 1
                temp1 = norm1[imax]
                temp2 = norm2[imax]
                for j in range(imax,nc):
                    norm1[j] = norm1[j+1]
                    norm2[j] = norm2[j+1]
                norm1[nc] = temp1
                norm2[nc] = temp2
            a = nc
            b = nfault
            prob[imult] = a/b
            if (imult > 0) and (prob[imult] < prob_max):
                break

            for j in range(imax,nf - nc):
                norm1[j] = norm1[j+nc]
                norm2[j] = norm2[j+nc]
            nc = nf-nc
            nf = nc
#                 
#             for icount in range(0,nf):
#                 print(icount)
#                 rota = np.empty(nc)
#                 norm1_avg, norm2_avg = mech_avg(nc, norm1, norm2)
#                 temp1 = norm1
#                 temp2 = norm2
#                 rota = np.array([mech_rot(norm1_avg,temp1[i],norm2_avg,temp2[i]) for i in range(0,nc)])[:,0]
# # #                 for i in range(0,nc):
# # # #                     print(norm1_avg,temp1[i],norm2_avg,temp2[i])
# # #                     rota[i], temp1[i], temp2[i] = mech_rot(norm1_avg,temp1[i],norm2_avg,temp2[i])
#                 maxrot = abs(rota).max()
#                 imax = np.where(rota == rota.max())[0][0]
#                 if maxrot <= cangle:
#                     break
#                 nc = nc - 1
#                 norm1 = np.delete(norm1,imax,axis=0)
#                 norm2 = np.delete(norm2,imax,axis=0)        
#             a = nc
#             b = nfault
#             prob[imult] = a / b
#             
#             if (imult > 0) and (prob[imult] < prob_max):
#                 break
#             nf = nc     
            
            rms_diff[0,imult] = 0
            rms_diff[1,imult] = 0
            
#             for i in range(nfault):
            temp1 = norm1in
            temp2 = norm2in
            
            test = np.hstack([np.broadcast_to(norm1_avg,(nfault,3)),temp1,np.broadcast_to(norm2_avg,(nfault,3)),temp2])
            rota, temp1, temp2 = mech_rot_2(test)
            d11 = np.dot(temp1,norm1_avg)
#                 print(d11)
            d22 = np.dot(temp2,norm2_avg)
            d11[d11 > 1] = 1
            d11[d11 > 1] = 1
            d22[d22 < -1] = -1
            d22[d22 < -1] = -1
            a11 = np.arccos(d11)
            a22 = np.arccos(d22)
            rms_diff[0,imult] = np.sum(a11 ** 2)
            rms_diff[1,imult] = np.sum(a22 ** 2)
            rms_diff[0,imult] = degrad * sqrt(rms_diff[0,imult]/nfault)
            rms_diff[1,imult] = degrad * sqrt(rms_diff[1,imult]/nfault)
                        
            str_avg[imult], dip_avg[imult], rak_avg[imult] = fpcoor_2SDR(norm1_avg,norm2_avg)
        nsltn = imult
 
    return nsltn, str_avg, dip_avg, rak_avg, prob, rms_diff

def get_misf_amp(npsta, p_azi_mc, p_the_mc, sp_ratio, p_pol, strike, dip, rake):
    rad = pi / 180
       
    p_azi = p_azi_mc[:,0]
    p_the = p_the_mc[:,0]
    
    M = np.empty([3,3])
    M[0,0] = -sin(dip) * cos(rake) * sin(2 * strike) - sin(2 * dip) * \
        sin(rake) * np.sin(strike)
    M[1,1] = sin(dip) * cos(rake) * sin(2 * strike) - sin(2 * dip) * \
        cos(rake) * cos(strike)
    M[2,2] = sin(2 * dip) * sin(rake)
    M[0,1] = sin(dip) * cos(rake) * cos(2 * strike) + 0.5 * sin(2 * dip) \
        * sin(rake) * sin(2 * strike)
    M[1,0] = M[0,1]
    M[0,2] = -cos(dip) * cos(rake) * cos(strike) - cos( 2 * dip) * sin(rake) \
        * sin(strike)
    M[2,0] = M[0,2]
    M[1,2] = -cos(dip) * cos(rake) * sin(strike) + cos(2 * dip) * sin(rake) \
        * cos(strike)
    M[2,1] = M[1,2]
    
    bb3, bb1 = fpcoor_2FS(strike,dip,rake)
    bb2 = cross(bb3,bb1)
    
    mfrac = 0
    qcount = 0
    stdr = 0
    scount = 0
    mavg = 0
    acount = 0
    
    for k in range(0, npsta):
        p_a1, p_a2, p_a3 = to_car(p_the[k], p_azi[k], 1)
        p_a = np.array([p_a1,p_a2,p_a3])
        p_b1 = np.dot(bb1,p_a)
        p_b3 = np.dot(bb3,p_a)
        p_proj1 = p_a1 - p_b3 * bb3[0]
        p_proj2 = p_a2 - p_b3 * bb3[1]
        p_proj3 = p_a3 - p_b3 * bb3[2]
        plen = sqrt(p_proj1 ** 2 + p_proj2 ** 2 + p_proj3 ** 2)
        p_proj1 = p_proj1 / plen
        p_proj2 = p_proj2 / plen
        p_proj3 = p_proj3 / plen
        p_proj = np.array([p_proj1,p_proj2,p_proj3])
        pp_b1 = np.dot(bb1,p_proj)
        pp_b2 = np.dot(bb2,p_proj)
        phi = atan2(pp_b2, pp_b1)
        theta = acos(p_b3)
        p_amp = abs(sin(2 * theta) * cos(phi))
        wt = sqrt(p_amp)
        if p_pol[k] != 0:
            azi = rad * p_azi[k]
            toff = rad * p_the[k]
            a = sin(toff) * cos(azi), sin(toff) * sin(azi), -cos(toff)
            b = np.zeros(3)
            for i in range(0,3):
                for j in range(0,3):
                    b[i] = b[i] + M[i,j] * a[j]
            if np.dot(a,b) < 0:
                pol = -1
            else:
                pol = 1
            if pol*p_pol[k] < 0:
                mfrac = mfrac + wt
            qcount = qcount + wt
            stdr = stdr + wt
            scount = scount + 1
        if sp_ratio[k] != 0 and not np.isnan(sp_ratio[k]):
            s1 = cos(2 * theta) * cos(phi)
            s2 = -cos(theta) * sin(phi)
            s_amp = sqrt(s1 ** 2 + s2 ** 2)
            sp_rat = np.log10(4.9 * s_amp / p_amp)
            mavg = mavg + abs(sp_ratio[k] - sp_rat)
            acount = acount + 1
            stdr = stdr + wt
            scount = scount + 1
    if qcount == 0:
        mfrac = 0
    else:
        mfrac = mfrac / qcount
    if acount == 0:
        mavg = 0
    else:
        mavg = mavg / acount
    if scount == 0:
        stdr = 0
    else:
        stdr = stdr / scount
    
    return mfrac, mavg, stdr
    
def main(test_events,test_data):
    nmc = 30 # Number of trials
    dang = 2 # minimum grid spacing
    ncoor = 472410 # number of test mechanisms
    maxout = 500 # Maximum focal mechanism outputs
    ratmin = 3 # Minimum allowed SNR
    badfrac = 0.1 # Fraction of polarities assumed bad
    qbadfrac = 0.3 # Assumed noise in amplitude ratios, log10
    cangle = 45
    prob_max = 0.1
    
    
    nrot, thetable, phitable, amptable, b1, b2, b3, astep, ntab = focal_arrays(ncoor, dang)
    
    foc_out = pd.DataFrame()
    for event in unique_events:
        # Create event Series
        test_ev = test_events[test_events.evid == event].iloc[0]
        # Create arrival dataframe
        test_event = test_data[test_data.evid == event].reset_index(drop=True)
        ev_data = test_ev.copy()
        arr_data = test_event.copy().reset_index(drop=True)
        icusp = arr_data.evid.unique()[0]
        sname = arr_data.sta.values
        qdist = arr_data.dist.values
        # Filter data based on qdist values... if greater than X km
    
        # Prepare P polarity data
        arr_data.loc[arr_data.fm.str.lower() == 'c','fm'] = 1
        arr_data.loc[arr_data.fm.str.lower() == 'u','fm'] = 1
        arr_data.loc[arr_data.fm == '+','fm'] = 1
        arr_data.loc[arr_data.fm.str.lower() == 'd','fm'] = -1
        arr_data.loc[arr_data.fm == '-','fm'] = -1
        arr_data.loc[arr_data.fm == '.','fm'] = 0
        arr_data.loc[arr_data.p_az < 0, 'p_az'] = arr_data.loc[arr_data.p_az < 0, 'p_az'] + 360
        
        pol_data = arr_data[arr_data.fm != 0][['fm','p_az','p_toa','sp_ratio']]
        pol_data['sp_ratio'] = 0
        nppl = len(pol_data)
        
        sp_data = arr_data[arr_data.sp_ratio != -1][['fm','p_az','p_toa','sp_ratio','p_amp','s_amp','p_noise','s_noise']]
        sp_data['fm'] = 0
        qcor = 0
        s2n1 = abs(sp_data.p_amp)/sp_data.p_noise
        s2n2 = abs(sp_data.s_amp)/sp_data.s_noise
        spin = sp_data.s_amp/abs(sp_data.p_amp)
        sp_amp = np.log10(spin) - qcor
        sp_data['sp_ratio'] = sp_amp
        sp_data.drop(columns=['p_amp','s_amp','p_noise','s_noise'],inplace=True)
        nspr = len(sp_data)
        
        arr_data = pd.concat([pol_data,sp_data]).reset_index(drop=True)
        npsta = len(arr_data)
        
        p_pol = arr_data.fm.values
        sp_amp = arr_data.sp_ratio.values
        qazi = arr_data.p_az.values
        qthe = arr_data.p_toa.values
#         qpamp = arr_data.p_amp.values
#         qsamp = arr_data.s_amp.values
#         qns1 = arr_data.p_noise.values
#         qns2 = arr_data.s_noise.values
    
        sazi = 1 # Azimuth uncertainty
        sthe = 1 # Takeoff angle uncertainty
        p_azi_mc = np.zeros([len(qazi), nmc])
        p_the_mc = np.zeros([len(qthe), nmc])
        p_azi_mc[:,0] = qazi
        p_the_mc[:,0] = qthe
        # Add random noise to azimuth and takeoff angle data
        for i in range(1, nmc):
            for j in range(0, len(qazi)):
                p_azi_mc[j,i] = qazi[j] + np.random.normal(sazi)
                p_the_mc[j,i] = qthe[j] + np.random.normal(sthe)
#         nppl = len(p_pol[p_pol != 0])
    
        # Prepare amplitude ratio data
#         qcor = 0 # S/P ratio correction term. Set to 0 for now.
#         s2n1 = abs(qpamp)/qns1
#         s2n2 = abs(qsamp)/qns2
#         spin = qsamp/abs(qpamp)
#         sp_amp = np.log10(spin) - qcor
# #         sp_amp = sp_ratio
#         nspr = len(sp_amp[~np.isnan(sp_amp)])
    #     if qpamp == 0:
    #         return
        nmismax = max(round(nppl*badfrac), 2)
        ntotal = nmismax
        nextra = max(round(nppl*badfrac*0.5), 2)
        qmismax = max(nspr * qbadfrac, 2.0)
        qtotal = qmismax
        qextra = max(nspr * qbadfrac*0.5, 2.0)

        print('Computing acceptable focal mechanism solutions for event '+str(icusp))
        nf, strike, dip, rake, faults, slips = focal_amp_mc(ncoor, nrot, thetable, phitable, 
            amptable, p_azi_mc, p_the_mc, sp_amp, p_pol, npsta, nmc, nextra, ntotal, 
            qextra, qtotal, maxout, arr_data, b1, b2, b3, astep, ntab)

        nout2 = min(maxout,nf)
        # Find probable mechanism from set of acceptable solutions
        print('Finding preferred mechanism for event '+str(icusp))
        nmult, str_avg, dip_avg, rak_avg, prob, var_est = mech_prob(cangle,prob_max,nout2,faults,slips)
    
        # Find misfit for preferred solution
        var_avg = np.zeros(nmult)
        qual = np.empty(nmult, dtype='str')
        for imult in range(0,nmult):
            var_avg[imult] = (var_est[0,imult] + var_est[1,imult]) / 2
            print('cid = ',icusp,imult,' mech = ',str_avg[imult],dip_avg[imult],rak_avg[imult])
            mfrac, mavg, stdr = get_misf_amp(npsta, p_azi_mc, p_the_mc, sp_amp, p_pol, 
                str_avg[imult], dip_avg[imult], rak_avg[imult])
    
            # Solution quality rating
            if prob[imult] > 0.8 and var_avg[imult] <= 25:
                qual[imult] = 'A'
            elif prob[imult] > 0.6 and var_avg[imult] <= 35:
                qual[imult] = 'B'
            elif prob[imult] > 0.5 and var_avg[imult] <= 45:
                qual[imult] = 'C'
            else:
                qual[imult] = 'D'
    
        focs = pd.DataFrame()
    
        for i in range(0,nmult):
            foc_data = ev_data.copy()
            if nmult > 1:
                mflag = '*'
            else:
                mflag = '.'
            foc_data['strike'] = str_avg[i]
            foc_data['dip'] = dip_avg[i]
            foc_data['rake'] = rak_avg[i]
            foc_data['fp_var'] = var_est[0,i] # Fault plane uncertainty in degrees
            foc_data['aux_var'] = var_est[1,i] # Aux plane uncertainty in degrees
            foc_data['prob'] = prob[i]
            foc_data['mfrac'] = mfrac * 100 # Weighted percent misfit polarities
            foc_data['mavg'] = mavg * 100 # Percent misfit average log10(s/P)
            foc_data['stdr'] = stdr # Station distribution ratio
            foc_data['nppl'] = nppl
            foc_data['nspr'] = nspr
            foc_data['qual'] = qual[i]
            foc_data['mflag'] = mflag
        
            focs = focs.append(foc_data)
        foc_out = foc_out.append(focs)
        
    return focs
            
test_events = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/earthquake_source_table_complete.csv',low_memory=False)
test_data = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/Scripts/Polarity/kaikoura_phase_table.csv',low_memory=False)

unique_events = test_data.evid.unique()