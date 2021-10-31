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
    degrad = 180/np.pi
    z = -r * cos(the * degrad)
    x = r * sin(the * degrad) * cos(phi*degrad)
    y = r * sin(the * degrad) * sin(phi*degrad)
    return x,y,z

def focal_amp_mc(ncoor,p_azi_mc, p_the_mc, sp_amp, sp_ratio, p_pol, npsta, nmc, dang, nextra, ntotal, 
    qextra, qtotal, maxout):
    #     ncoor = 729000
    dangold = -1
    ntab = 180
    degrad = 180/np.pi
    rad = 1/degrad
    b1 = np.empty([3, ncoor])
    b2 = np.empty([3, ncoor])
    b3 = np.empty([3, ncoor])
    thetable = np.empty(2*ntab+1)
    phitable = np.empty([2*ntab+1,2*ntab+1])
    amptable = np.empty([2,ntab+1,2*ntab+1])
    qmis = np.empty(ncoor)
    nmis = np.empty(ncoor)
    irotgood = np.empty(ncoor,dtype='int')
    irotgood2 = np.empty(ncoor,dtype='int')

    try:
        if dangold != dang:
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
        dangold=dang

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
        for irot in range(0,nrot): # IS THIS NECESSARY?
            irotgood[irot] = 0


        # loop over multiple trials, this is the slowest part of the process
        for im in range(0,nmc):
            nmis0min=1e5
            qmis0min=1.0e5
            qmis = np.zeros(irot)
            nmis = np.zeros(irot)
            p_a1 = np.empty(npsta)
            p_a2 = np.empty(npsta)
            p_a3 = np.empty(npsta)
            # Convert data to Cartesian coordinates
            for idx, arr in test_event.iterrows():
                p_a1[idx], p_a2[idx], p_a3[idx] = to_car(arr.p_toa, arr.p_az, 1)
            #  Find misfit for each solution and minimum misfit
            nmis0min=1e5
            qmis0min=1.0e5
            qmis = np.zeros(irot)
            nmis = np.zeros(irot)
            p_a = np.array([p_a1,p_a2,p_a3])
            p_b1 = p_a.T @ b1
            p_b3 = p_a.T @ b3
            p_proj1 = p_a1.reshape(npsta,1) - p_b3 * b3[0, :]
            p_proj2 = p_a2.reshape(npsta,1) - p_b3 * b3[1, :]
            p_proj3 = p_a3.reshape(npsta,1) - p_b3 * b3[2, :]
            plen = np.sqrt(p_proj1 ** 2 + p_proj2 ** 2 + p_proj3 ** 2)
            p_proj1 = p_proj1/plen
            p_proj2 = p_proj2/plen
            p_proj3 = p_proj3/plen
            pp_b1 = b1[0, :] * p_proj1 + b1[1,:] * p_proj2 + b1[2, :] * p_proj3
            pp_b2 = b2[0, :] * p_proj1 + b2[1,:] * p_proj2 + b2[2, :] * p_proj3
            i = np.round((p_b3+1)/astep).astype('int')
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
            sp_ratio[p_amp == 0] = 4
            sp_ratio[s_amp == 0] = -2
            sp_ratio_mask = (p_amp != 0) & (s_amp != 0)
            sp_ratio[sp_ratio_mask] = np.log10(4.9 * s_amp[sp_ratio_mask] / p_amp[sp_ratio_mask])
            qmis = np.nansum(abs(sp_amp.reshape(npsta,1) - sp_ratio),axis=0)
            p_pol_mask = p_pol != 0
            prod = p_b1[p_pol_mask] * p_b3[p_pol_mask]
            ipol = np.full(prod.shape,-1)
            ipol[prod > 0] = 1
            p_pol_compare = p_pol[p_pol_mask].reshape(len(ipol),1)
            nmis = np.sum(ipol != p_pol_compare,axis=0)
            nmis0min = nmis.min()
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

def mech_rot(norm1,norm2,slip1,slip2):
    degrad = 180/np.pi
    rotemp = np.empty(4)
    for itr in range(0,4):
        n1 = np.empty(3)
        n2 = np.empty(3)
        if itr < 2:
            norm2_temp = norm2
            slip2_temp = slip2
        else:
            norm2_temp = slip2
            slip2_temp = norm2
        if (itr == 1) | (itr == 3):
            norm2_temp = -norm2_temp
            slip2_temp = -slip2_temp
        
        B1 = cross(norm1, slip1)
        B2 = cross(norm2_temp, slip2_temp)
        
        phi1 = np.dot(norm1,norm2_temp)
        phi2 = np.dot(slip1,slip2_temp)
        phi3 = np.dot(B1,B2)
        phi = np.arccos(phi1), np.arccos(phi2), np.arccos(phi3)
        
        # If the mechanisms are very close, rotation = 0
        if (phi[0] < 1e-4) and (phi[1] < 1e-4) and (phi[2] < 1e-4):
            rotemp[itr] = 0
        # If one vector is the same, it is the rotation axis
        elif phi[0] < 1e-4:
            rotemp[itr] = degrad * phi[1]
        elif phi[1] < 1e-4:
            rotemp[itr] = degrad * phi[2]
        elif phi[2] < 1e-4:
            rotemp[itr] = degrad * phi[0]
        # Find difference vectors - the rotation axis must be orthogonal to all three of
        # these vectors
        else:
            n = np.array([np.array(norm1) - np.array(norm2_temp), np.array(slip1) - np.array(slip2_temp), np.array(B1) - np.array(B2)])
            scale = np.sqrt(np.sum(n ** 2,axis=0))
            n = n / scale
            qdot = n[0][1] * n[0][2] + n[1][1] * n[1][2] + n[2][1] * n[2][2], \
                n[0][0] * n[0][2] + n[1][0] * n[1][2] + n[2][0] * n[2][2], \
                n[0][0] * n[0][1] + n[1][0] * n[1][1] + n[2][0] * n[2][1]
                
            # Use the two largest difference vectors, as long as they aren't orthogonal
            iout = -1
            for i in range(0,3):
                if qdot[i] > 0.9999:
                    iout = i
            if iout == -1:
                qmins = scale.min()
                iout = np.where(scale == scale.min())[0][0]
            k = 1
            for j in range(0,3):
                if j != iout:
                    if k == 1:
                        for i in range(0,3):
                            n1[i] = n[i,j]
                        k = 2
                    else:
                        for i in range(0,3):
                            n2[i] = n[i,j]
            # Find rotation axis by taking cross product
            R = cross(n1,n2)
            scaleR = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
#             scaleR = np.sqrt(np.sum(R ** 2))
            R = R / scaleR
            
            # Find rotation using axis furthest from rotation axis
            theta = np.array([np.arccos(np.dot(norm1,R)), \
                np.arccos(np.dot(slip1,R)), \
                np.arccos(np.dot(B1,R))])
            qmindifs = abs(theta - np.pi/2)
            qmindif = qmindifs.min()
            iuse = np.where(qmindifs == qmindif)[0][0]
            rotemp[itr] = (np.cos(phi[iuse]) - np.cos(theta[iuse]) * np.cos(theta[iuse]) \
                ) / (np.sin(theta[iuse]) * np.sin(theta[iuse]))
            if rotemp[itr] > 1:
                rotemp[itr] = 1
            if rotemp[itr] < -1:
                rotemp[itr] = -1
            rotemp[itr] = degrad * np.arccos(rotemp[itr])
    # Find the minimum rotation for the 4 combos and change norm2 and slip2
        rota = 180
        rota = abs(rotemp).min()
        irot = np.where(abs(rotemp) == rota)[0][0]
        if irot >= 2:
            qtemp = slip2
            slip2 = norm2
            norm2 = qtemp
        if (irot == 1) | (irot == 3):
            norm2 = -norm2
            slip2 = -slip2         
    return rota, norm2, slip2

def mech_avg(nf,norm1,norm2):
    degrad = 180/np.pi
    
    # If there is only one mechanism, return that mechanism
    if nf <= 1:
        norm1_avg = norm1
        norm2_avg = norm2
    else:
        norm1_avg = norm1[0]
        norm2_avg = norm2[0]
        ref1 = norm1[0]
        ref2 = norm2[0]
        avang1 = 0
        avang2 = 0
        for i in range(1,nf):
            temp1 = norm1[i]
            temp2 = norm2[i]
            rota, temp1, temp2 = mech_rot(ref1,temp1,ref2,temp2)
            norm1_avg = norm1_avg + temp1
            norm2_avg = norm2_avg + temp2
        ln_norm1 = np.sqrt(np.sum(norm1_avg * norm1_avg))
        ln_norm2 = np.sqrt(np.sum(norm2_avg * norm2_avg))
        norm1_avg = norm1_avg / ln_norm1
        norm2_avg = norm2_avg / ln_norm2
        
        for i in range(1,nf):
            temp1 = norm1[i]
            temp2 = norm2[i]
            rota, temp1, temp2 = mech_rot(norm1_avg,temp1,norm2_avg,temp2)
            d11 = np.dot(temp1,norm1_avg)
            d22 = np.dot(temp2,norm2_avg)
            if d11 >= 1:
                d11 = 1
            elif d11 <= -1:
                d11 = -1
            if d22 >= 1:
                d22 = 1
            elif d22 <= -1:
                d22 = -1
            a11 = np.arccos(d11)
            a22 = np.arccos(d22)
            avang1 = avang1 + a11 * a11
            avang2 = avang2 + a22 * a22
        avang1 = np.sqrt(avang1/nf)
        avang2 = np.sqrt(avang2/nf)
        
        # The average normal vectors may not be exactly orthogonal (although they are
        # usually very close) - find the misfit from orthogonal and adjust the vectors to
        # to make them orthogonal - adjust the more poorly constrained plane more
        
        if avang1 + avang2 >= 0.0001:
            maxmisf = 0.01
            fract1 = avang1/(avang1 + avang2)
            for icount in range(0,100):
                dot1 = np.dot(norm1_avg,norm2_avg)
                misf = 90 - np.arccos(dot1) * degrad
                if abs(misf) <= maxmisf:
                    break
                theta1 = misf * fract1 / degrad
                theta2 = misf * (1 - fract1) / degrad
                temp = norm1_avg
                norm1_avg = norm1_avg - norm2_avg * np.sin(theta1)
                norm2_avg = norm2_avg - temp * np.sin(theta2)
                ln_norm1 = np.sqrt(np.sum(norm1_avg ** 2))
                ln_norm2 = np.sqrt(np.sum(norm2_avg ** 2))
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
                nsltn = imult # Use only ones with certain probability
                break
#             for icount in range(0,nf):
#                 print(icount)
#                 rota = np.empty(nc)
#                 norm1_avg, norm2_avg = mech_avg(nc, norm1, norm2)
#                 temp1 = norm1
#                 temp2 = norm2
#                 for i in range(0,nc):
#                     print(norm1_avg,temp1[i],norm2_avg,temp2[i])
#                     rota[i], temp1[i], temp2[i] = mech_rot(norm1_avg,temp1[i],norm2_avg,temp2[i])
#                 maxrot = rota.max()
#                 imax = np.where(rota == rota.max())[0][0]
#                 if maxrot <= cangle:
#                     break
#                 nc = nc - 1
#                 temp1 = norm1[imax]
#                 temp2 = norm2[imax]
#                 for j in range(imax,nc):
#                     norm1[j] = norm1[j+1]
#                     norm2[j] = norm2[j+1]
#             a = nc
#             b = nfault
#             prob[imult] = a/b
#             if (imult > 0) and (prob[imult] < prob_max):
#                 break
# 
#             for i in range(0,nf - nc):
#                 norm1[j] = norm1[j+nc]
#                 norm2[j] = norm2[j+nc]
#             nc = nf-nc
#             nf = nc
                
            for icount in range(0,nf):
                print(icount)
                rota = np.empty(nc)
                norm1_avg, norm2_avg = mech_avg(nc, norm1, norm2)
                temp1 = norm1
                temp2 = norm2
                for i in range(0,nc):
                    print(norm1_avg,temp1[i],norm2_avg,temp2[i])
                    rota[i], temp1[i], temp2[i] = mech_rot(norm1_avg,temp1[i],norm2_avg,temp2[i])
                maxrot = rota.max()
                imax = np.where(rota == rota.max())[0][0]
                if maxrot <= cangle:
                    break
                nc = nc - 1
                norm1 = np.delete(norm1,imax,axis=0)
                norm2 = np.delete(norm2,imax,axis=0)
            a = nc
            b = nfault
            prob[imult] = a / b
            
            if (imult > 0) and (prob[imult] < prob_max):
                break
            
#             nc = nf - nc
            nc = nf - nc
            nf = nc
            
            rms_diff[0,imult] = 0
            rms_diff[1,imult] = 0
            
            for i in range(0,nfault):
                temp1 = norm1in[i]
                temp2 = norm2in[i]
                rota = mech_rot(norm1_avg,temp1,norm2_avg,temp2)
                d11 = np.dot(temp1,norm1_avg)
                d22 = np.dot(temp2,norm2_avg)
                if d11 >= 1:
                    d11 = 1
                elif d11 <= -1:
                    d11 = -1
                if d22 >= 1:
                    d22 = 1
                elif d22 <= -1:
                    d22 = -1
                a11 = np.arccos(d11)
                a22 = np.arccos(d22)
                rms_diff[0,imult] = rms_diff[0,imult]+a11*a11
                rms_diff[1,imult] = rms_diff[1,imult]+a22*a22
            rms_diff[0,imult] = degrad*np.sqrt(rms_diff[0,imult]/nfault)
            rms_diff[1,imult] = degrad*np.sqrt(rms_diff[1,imult]/nfault)
            
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
    
def main(test_ev,test_event):
    nmc = 30 # Number of trials
    dang = 2 # minimum grid spacing
    dang2 = dang
    ncoor = 472410 # number of test mechanisms
    maxout = 500 # Maximum focal mechanism outputs
    ratmin = 3 # Minimum allowed SNR
    badfrac = 0.1 # Fraction of polarities assumed bad
    qbadfrac = 0.3 # Assumed noise in amplitude ratios, log10
    cangle = 45
    prob_max = 0.1
    
    
    foc_out = pd.DataFrame()
    for event in unique_events[10:20]:
        test_ev = test_events[test_events.evid == event].iloc[0]
        test_event = test_data[test_data.evid == event].reset_index(drop=True)
        ev_data = test_ev.copy()
        arr_data = test_event.copy().reset_index(drop=True)
        icusp = arr_data.evid.unique()[0]
        sname = arr_data.sta.values
        qdist = arr_data.dist.values
        # Filter data based on qdist values... if greater than X km
        npsta = len(arr_data)
    
        # Prepare P polarity data
        arr_data.loc[arr_data.fm.str.lower() == 'c','fm'] = 1
        arr_data.loc[arr_data.fm.str.lower() == 'u','fm'] = 1
        arr_data.loc[arr_data.fm == '+','fm'] = 1
        arr_data.loc[arr_data.fm.str.lower() == 'd','fm'] = -1
        arr_data.loc[arr_data.fm == '-','fm'] = -1
        arr_data.loc[arr_data.fm == '.','fm'] = 0
        arr_data.loc[arr_data.p_az < 0, 'p_az'] = arr_data.loc[arr_data.p_az < 0, 'p_az'] + 360
        p_pol = arr_data.fm.values
        qazi = arr_data.p_az.values
        qthe = arr_data.p_toa.values
        qpamp = arr_data.p_amp.values
        qsamp = arr_data.s_amp.values
        qns1 = arr_data.p_noise.values
        qns2 = arr_data.s_noise.values
    
        sazi = 1 # Azimuth uncertainty
        sthe = 1 # Takeoff angle uncertainty
        p_azi_mc = np.zeros([len(qazi), nmc])
        p_the_mc = np.zeros([len(qthe), nmc])
        p_azi_mc[:,0] = qazi
        p_the_mc[:,0] = qthe
        for i in range(1, nmc):
            for j in range(0, len(qazi)):
                p_azi_mc[j,i] = qazi[j] + np.random.normal(sazi)
                p_the_mc[j,i] = qthe[j] + np.random.normal(sthe)
        nppl = len(p_pol[p_pol != 0])
    
        # Prepare amplitude ratio data
        qcor = 0 # S/P ratio correction term. Set to 0 for now.
        s2n1 = abs(qpamp)/qns1
        s2n2 = abs(qsamp)/qns2
        spin = qsamp/abs(qpamp)
        sp_ratio = np.log10(spin) - qcor
        sp_amp = sp_ratio
        nspr = len(sp_ratio[~np.isnan(sp_ratio)])
    #     if qpamp == 0:
    #         return
        nmismax = max(round(nppl*badfrac), 2)
        ntotal = nmismax
        nextra = max(round(nppl)*badfrac, 2)
        qmismax = max(nspr * qbadfrac, 2.0)
        qtotal = qmismax
        qextra = max(nspr * qbadfrac, 2.0)

        print('Computing acceptable focal mechanism solutions for event '+str(icusp))
        nf, strike, dip, rake, faults, slips = focal_amp_mc(ncoor, p_azi_mc, p_the_mc, sp_amp, 
            sp_ratio, p_pol, npsta, nmc, dang, nextra, ntotal, qextra, qtotal, maxout)

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
            mfrac, mavg, stdr = get_misf_amp(npsta, p_azi_mc, p_the_mc, sp_ratio, p_pol, 
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
            foc_data['mflag'] = mflag
        
            focs = focs.append(foc_data)
        foc_out.append(focs)
        
    return focs
            
test_events = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/earthquake_source_table_complete.csv',low_memory=False)
test_data = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/Scripts/Polarity/kaikoura_phase_table.csv',low_memory=False)

unique_events = test_data.evid.unique()