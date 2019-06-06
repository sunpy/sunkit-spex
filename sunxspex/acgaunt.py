import numpy as np 
import astropy.units as u

class Acgaunt():

    """
    Class for calculating continuum gaunt factor using approximations
    of R. Mewe (18-JUN-85) to full calculations of 
    paper VI (Arnaut and Rothenflug for ion balances).
    
    """

    @u.quantity_input(wavelength=u.angstrom, temperature=u.K)
    def __init__(self, wavelength, temperature):
        '''
        Initializing required variables for Gaunt factor calculations

        Parameters
        ----------
        wavelength : `~astropy.Quantity`
            Wavelength in angstrom (1-d vector or scaler)
        temperature : `u.Quantity`
            Temperature 

        Notes
        -----
        The local variables are also initialized here

        '''

        self.te_6 = temperature.to_value() / 1e6 
        self.wave =wavelength.to_value() 
        # label local variables 
        self.Lambda, self.temp, self.CC, self.DD = setup_variables()

        # label local variables for the Gaunt free-bound calculation
        self.al, self.bl, self.cl, self.dl, self.tem_liml, self.wav_liml = setup_freebound_low()
        self.a, self.b, self.c, self.d, self.tem_lim, self.wav_lim = setup_freebound_high()


        # put data into array form if not already
        if isinstance(self.te_6, (int, float)):
            self.te_6 = np.array([self.te_6])
            self.nte_6 = len(self.te_6)
        else:
            self.te_6 = np.array(self.te_6)
            self.nte_6 = len(self.te_6)

        if isinstance(self.wave, (int, float)):
            self.wave = np.array([self.wave])
            self.nwave = len(self.wave)
        else:
            self.nwave = len(self.wave)


    def gaunt_ff(self):
        '''
        Function to calculate free-free Gaunt factor 

        Notes
        -----
        Good for variables in the ranges (1e4 < te_6 < 1e6) and (1< wave <1000)
        '''

        gaunt_ff=np.zeros((self.nte_6, self.nwave))

        low = np.where(self.te_6 <=  1.)[0] # low temperature  
        high = np.where(self.te_6 >  1.)[0] # high temperature

        if len(low) > 0:
            dummy=_mkdarr(self.wave,self.te_6[low])
            te_dummy=dummy[:, 1]
            wave_dummy = dummy[:,0].clip(min = 1., max = 1000)

            gaunt_ff[low,0:self.nwave-1]=0.29*wave_dummy**(0.48*(wave_dummy**(-0.08))) * te_dummy**(0.133*np.log10(wave_dummy)-0.2)  
            

        if len(high) > 0:
            dummy=_mkdarr(self.wave,self.te_6[high])
            te_dummy=dummy[:, 1]
            wave_dummy=dummy[:, 0]
            gaunt_ff[high,0:self.nwave]=1.01*wave_dummy**(0.355*(wave_dummy**(-0.06))) * (te_dummy/100.)**(0.3*(wave_dummy**(-0.066)))   

        return gaunt_ff


    def gaunt_2p(self):
        '''
        Function to calculate 2-photon Gaunt factor 

        '''
        gaunt_2p=np.zeros((self.nte_6, self.nwave))

        finit_2p = np.arange(len(self.wave))[np.logical_and(self.wave>19.,self.wave<=200)]

        if len(finit_2p) > 0: 
            dum_2p=_mkdarr(self.wave[finit_2p],self.te_6)
            te_2p=dum_2p[:, 1]
            wv_2p=dum_2p[:, 0]
            for i in range(0, 6):  
                alpha = 106. / (self.lamda[i]) * (te_2p**(-0.94))
                gaunt_2p[0:self.nte_6,finit_2p]=gaunt_2p[0:self.nte_6,finit_2p] + (wv_2p > self.lamda[i]) \
                                            * self.CC[i]  * ((self.lamda[i]/wv_2p)**alpha) \
                                            * np.sqrt(np.abs(np.cos(np.pi*((self.lamda[i]/wv_2p)-0.5)))) \
                                            * ((self.temp[i] / te_2p)**0.45) \
                                            * 10.**((-self.DD[i]*(np.log10(te_2p/self.temp[i]))**2)>(-37))    


        return gaunt_2p


    def gaunt_fb(self):
        '''
        Function to calculate free-bound Gaunt factor 

        '''
        
        gaunt_fb=np.zeros((self.nte_6, self.nwave))

        low = np.arange(len(self.te_6))[np.logical_and(self.te_6 >= 0.01, self.te_6 < 0.1)]
        high = np.where(self.te_6 >= 0.1)[0]

        # Low temperature case
        if len(low) > 0: 

            il = _indd(self.tem_liml,self.te_6[low])  
            jl = _indd(self.wav_liml,self.wave)        

            ijl=_mkdarr(jl,il)

            il=ijl[:,1]
            jl=ijl[:,0]
            

            tel=np.resize(np.resize(self.te_6[low],(len(self.te_6[low]), len(self.wave))),
                         len(self.te_6[low])*len(self.wave), 1)



            gaunt_fb[low,0:self.nwave] = self.al[il,jl]*(tel**self.bl[il,jl])  \
                                    * np.exp(self.cl[il,jl]*(tel**self.dl[il,jl]))

        # high temperature case
        if len(high) > 0: 
      
            i = _indd( self.tem_lim,self.te_6[high])  
            j = _indd( self.wav_lim,self.wave )        


            ij=_mkdarr(j,i)

            i=ij[:,1]
            j=ij[:,0]

            teh=np.reshape(np.resize(self.te_6[high],(len(self.te_6[high]),len(self.wave))), \
                           len(self.te_6[high])*len(self.wave), 1)
                 

            gaunt_fb[high,0:self.nwave] = self.a[i,j]*(teh**self.b[i,j])  \
                                    * np.exp(self.c[i,j]*(teh**self.d[i,j]))

        return gaunt_fb



    def acgaunt(self):
        '''
        Function to return the calculated continuum gaunt factor such as returned from acgaunt.pro
        '''
        return self.gaunt_ff() + self.gaunt_2p() + self.gaunt_fb()



#---------------------#
# These can move 
#---------------------#

def setup_variables():
    '''
     Setup of variables used in the calculations
    '''

    # OVIII, OVII, NVII,  NVI ,  CVI,  CV
    lamda = [19,   22,   25,   29,   34,   41]    # Wavelength of edges
    temp  = [2.45,  0.9,  1.7,  0.6, 1.05, 0.37]  # Max temperature of edges
    CC    = [  3.2, 11.3, 0.69,  2.1,  3.6, 10.3] # Noramlization coefficient
    DD    = [ 10.3,  2.5, 10.3,  2.5, 10.3,  2.5] # exponent coefficient
    return lamda, temp, CC, DD



def setup_freebound_low():
    ''' 
    Function to return the Local variables for free-bound 
    calculation for low temperatures.

    There are 5 temperature regions and 4 wavelength regions
    '''

    # Normalization Coefficients
    al = np.array([[2.48e-01, 2.48e-01, 2.48e-01, 5.35e-02],
                   [5.42e-12, 5.42e-12, 5.42e-12, 5.35e-02],
                   [3.68e-04, 3.68e-04, 3.23e-01, 5.35e-02],
                   [1.86e+03, 1.76e-01, 3.23e-01, 5.35e-02],
                   [6.50e-03, 1.76e-01, 3.23e-01, 5.35e-02]])

    # Exponent Coefficients
    bl = np.array([[-1.   , -1.   , -1.   , -1.   ],
                   [-9.39 , -9.39 , -9.39 , -1.   ],
                   [-4.9  , -4.9  , -1.   , -1.   ],
                   [-0.686, -1.   , -1.   , -1.   ],
                   [-5.41 , -1.   , -1.   , -1.   ]])

    # Exponent Coefficients
    cl = np.array([[0.158, 0.158, 0.158, 0.046],
                   [0.   , 0.   , 0.   , 0.046],
                   [0.   , 0.   , 0.16 , 0.046],
                   [0.   , 0.233, 0.16 , 0.046],
                   [0.   , 0.233, 0.16 , 0.046]])

    # Exponent Coefficients
    dl = np.array([[-1., -1., -1., -1.],
                   [ 0.,  0.,  0., -1.],
                   [ 0.,  0., -1., -1.],
                   [ 0., -1., -1., -1.],
                   [ 0., -1., -1., -1.]])


    tem_liml=[.015, .018, .035, .07]  # Boundaries between temperature bins
    wav_liml=[227.9, 504.3, 911.9]    # Boundaries btwn wavelength  bins
    return al, bl, cl, dl, tem_liml, wav_liml 

def setup_freebound_high():
    '''
    Function to return the local variables for free-bound 
    calculation for high temperatures
 
    There are  10 temperature regions and 16 wavelength regions
    '''

    # Normalization Coefficients
    a = np.array([[ 0.68 ,  0.68 ,  0.68 ,  0.68 ,  0.68 ,  0.68 ,  0.68 ,  0.68 ,
                    0.68 ,  0.68 ,  0.68 ,  0.68 ,  0.68 ,  0.6  ,  0.37 ,  0.053],
                  [ 3.73 ,  3.73 ,  3.73 ,  3.73 ,  3.73 ,  3.73 ,  3.73 ,  3.73 ,
                    3.73 ,  3.73 ,  3.73 ,  3.73 ,  3.73 ,  0.6  ,  0.37 ,  0.053],
                  [ 5.33 ,  5.33 ,  5.33 ,  5.33 ,  5.33 ,  5.33 ,  5.33 ,  5.33 ,
                    5.33 ,  5.33 ,  0.653,  0.653,  0.653,  0.6  ,  0.37 ,  0.053],
                  [ 14.  , 14.   , 14.   , 14.   , 14.   , 14.   , 14.   , 14.   ,
                    10.2 , 10.2  ,  1.04 ,  0.653,  0.653,  0.6  ,  0.37 ,  0.053],
                  [ 49.  , 49.   , 49.   , 49.   , 49.   , 49.   , 12.3  , 12.3  ,
                    10.2 ,  3.9  ,  1.04 ,  0.653,  0.653,  0.6  ,  0.37 ,  0.053],
                  [ 49.  , 49.   , 49.   , 49.   , 22.4  , 22.4  , 12.3  , 12.3  ,
                    10.2 ,  3.9  ,  1.04 ,  0.653,  0.653,  0.6  ,  0.37 ,  0.053],
                  [ 49.  , 49.   , 49.   , 49.   , 22.4  , 46.3  , 12.3  , 10.2  ,
                    10.2 ,  2.04 ,  1.04 ,  1.04 ,  0.653,  0.6  ,  0.37 ,  0.053],
                  [ 4.2  ,  4.2  ,  5.08 ,  3.75 ,  2.12 , 46.3  , 12.3  , 10.2  ,
                    10.2 ,  2.04 ,  1.04 ,  1.04 ,  0.653,  0.6  ,  0.37 ,  0.053],
                  [ 4.2  ,  4.2  ,  5.08 ,  3.75 ,  2.12 ,  6.12 ,  6.12 ,  4.98 ,
                    4.98 ,  1.1  ,  1.04 ,  1.04 ,  0.653,  0.6  ,  0.37 ,  0.053],
                  [ 4.2  , 18.4  , 18.4  ,  3.75 ,  2.12 ,  6.12 ,  6.12 ,  4.98 ,
                    4.98 ,  1.1  ,  1.04 ,  1.04 ,  0.653,  0.6  ,  0.37 ,  0.053]])

    # Exponent Coefficients
    b = np.array([[-1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ,
                   -1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                  [-1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ,
                   -1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                  [-1.595, -1.595, -1.595, -1.595, -1.595, -1.595, -1.595, -1.595,
                  -1.595, -1.595, -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                  [-0.543, -0.543, -0.543, -0.543, -0.543, -0.543, -0.543, -0.543,
                  -2.19 , -2.19 , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                 [-1.572, -1.572, -1.572, -1.572, -1.572, -1.572, -2.09 , -2.09 ,
                  -2.19 , -2.763, -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                 [-1.572, -1.572, -1.572, -1.572, -1.2  , -1.2  , -2.09 , -2.09 ,
                  -2.19 , -2.763, -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                 [-1.572, -1.572, -1.572, -1.572, -1.2  , -3.06 , -2.09 , -2.19 ,
                  -2.19 , -1.31 , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                 [-0.82 , -0.82 , -1.   , -1.   , -1.   , -3.06 , -2.09 , -2.19 ,
                  -2.19 , -1.31 , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                 [-0.82 , -0.82 , -1.   , -1.   , -1.   , -1.556, -1.556, -1.556,
                  -1.556, -1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ],
                 [-0.82 , -1.33 , -1.33 , -1.   , -1.   , -1.556, -1.556, -1.556,
                  -1.556, -1.   , -1.   , -1.   , -1.   , -1.   , -1.   , -1.   ]])

    # Exponent Coefficients
    c = np.array([ [ 0.55 ,  0.55 ,  0.55 ,  0.55 ,  0.55 ,  0.55 ,  0.55 ,  0.55 ,
                    0.55 ,  0.55 ,  0.55 ,  0.55 ,  0.55 ,  0.55 ,  0.158,  0.05 ],
                   [ 0.21 ,  0.21 ,  0.21 ,  0.21 ,  0.21 ,  0.21 ,  0.21 ,  0.21 ,
                     0.21 ,  0.21 ,  0.21 ,  0.21 ,  0.21 ,  0.55 ,  0.158,  0.05 ],
                   [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
                     0.   ,  0.   ,  0.72 ,  0.72 ,  0.72 ,  0.55 ,  0.158,  0.05 ],
                   [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
                    -0.208, -0.208,  0.58 ,  0.72 ,  0.72 ,  0.55 ,  0.158,  0.05 ],
                   [-0.826, -0.826, -0.826, -0.826, -0.826, -0.826, -0.208, -0.208,
                    -0.208,  0.   ,  0.58 ,  0.72 ,  0.72 ,  0.55 ,  0.158,  0.05 ],
                   [-0.826, -0.826, -0.826, -0.826,  0.   ,  0.   , -0.208, -0.208,
                    -0.208,  0.   ,  0.58 ,  0.72 ,  0.72 ,  0.55 ,  0.158,  0.05 ],
                   [-0.826, -0.826, -0.826, -0.826,  0.   ,  0.   , -0.208, -0.208,
                    -0.208,  0.   ,  0.58 ,  0.58 ,  0.72 ,  0.55 ,  0.158,  0.05 ],
                   [ 4.   ,  4.   ,  3.9  ,  4.2  ,  5.6  ,  0.   , -0.208, -0.208,
                    -0.208,  0.   ,  0.58 ,  0.58 ,  0.72 ,  0.55 ,  0.158,  0.05 ],
                   [ 4.   ,  4.   ,  3.9  ,  4.2  ,  5.6  ,  0.   ,  0.   ,  0.   ,
                     0.   ,  0.58 ,  0.58 ,  0.58 ,  0.72 ,  0.55 ,  0.158,  0.05 ],
                   [ 4.   ,  0.   ,  0.   ,  4.2  ,  5.6  ,  0.   ,  0.   ,  0.   ,
                     0.   ,  0.58 ,  0.58 ,  0.58 ,  0.72 ,  0.55 ,  0.158,  0.05 ]])


    # Exponent Coefficients
    d = np.array([ [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                    -1., -1., -1.],
                   [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                    -1., -1., -1.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.,
                    -1., -1., -1.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -2., -2., -1., -1., -1.,
                    -1., -1., -1.],
                   [-1., -1., -1., -1., -1., -1., -2., -2., -2.,  0., -1., -1., -1.,
                    -1., -1., -1.],
                   [-1., -1., -1., -1.,  0.,  0., -2., -2., -2.,  0., -1., -1., -1.,
                    -1., -1., -1.],
                   [-1., -1., -1., -1.,  0.,  0., -2., -2., -2.,  0., -1., -1., -1.,
                    -1., -1., -1.],
                   [-1., -1., -1., -1., -1.,  0., -2., -2., -2.,  0., -1., -1., -1.,
                    -1., -1., -1.],
                   [-1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1., -1., -1., -1.,
                    -1., -1., -1.],
                   [-1.,  0.,  0., -1., -1.,  0.,  0.,  0.,  0., -1., -1., -1., -1.,
                    -1., -1., -1.]])

    
    tem_lim = [.2,.258,.4,.585,1.,1.5,3.,4.5,8.] # Boundaries between temperature bins
    wav_lim = [1.4,4.6,6.1,9.1,14.2,16.8,18.6,22.5,25.3,31.6,51.9,57.0,89.8,227.9,911.9]  # Boundaries between wavelength bins

    return a, b, c, d, tem_lim, wav_lim



# rough and ready mkdarr function from /ssw/gen/idl/genutils/mkdarr.pro
def _mkdarr(a, b):
    if isinstance(a, (int, float)):
        a = np.array([a])
    if isinstance(b, (int, float)):
        b = np.array([b])


    new_arr = []
    for i in range(len(a)):
        for j in range(len(b)):
            new_arr.append([a[i], b[j]])

    return np.array(new_arr)

#another idl code, probably a python version somewhere but for moment just copied from IDL to python
def _indd(a, b):

    if isinstance(a, (int, float)):
        a = np.array([a])
    if isinstance(b, (int, float)):
        b = np.array([b])

    npts = len(a)
    result = np.zeros(len(b), dtype = 'int')

    ii = np.where(b > a[npts-1])[0]
    nn1 = len(ii)
    if nn1 > 0:
        result[ii] = npts

    ii = np.arange(len(b))[np.logical_and(b > a[0], b <= a[npts-1])]
    nn1 = len(ii)
    if nn1 > 0: 
        for i in range(0, nn1):
            result[ii[i]] = np.max(np.where(b[ii[i]] > a)[0]) + 1

    return result
