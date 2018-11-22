
# Functions to load delay data produced by the raytracer

import numpy as np
from numpy import exp

Planck_constant = 6.62606979e-34
light_speed = 299792458.0



def load_raw(fname):
    # Read numerical data
    raw_data = np.loadtxt(fname, skiprows=7)

    # Read metadata. Wavelength is converted from nanometers to meters for future computations.
    metadata = {}
    with open(fname, "r") as f:
        f.readline()
        metadata["wavelength"] = float(f.readline().split()[3]) * 1e-9
        metadata["tmin"] = float(f.readline().split()[3])
        metadata["tmax"] = float(f.readline().split()[3])
    return raw_data, metadata


def scale_data(raw_data, metadata, geometry):
    """Convert raw data to photon counts, given beam and detector properties."""
    # Time delay axis in seconds
    X = np.linspace(metadata["tmin"], metadata["tmax"], len(raw_data)) / light_speed
    delta = X[1] - X[0]

    # Raw data has units of m^2 / steradian, disk-integrated brightness at given time delay
    # for 1 Watt / m^2 incident radiation, integrated over bin.
    data = raw_data.copy()

    # Dividing data by bin width, get m^2 / steradian / s.
    data /= delta

    # Incident flux in Joules per square meter
    beam_flux = geometry["beam_energy"] / geometry["beam_cross_section"] 
    
    # Number of photons in one Joule of energy at this wavelength.
    photons_per_joule = metadata["wavelength"] / (Planck_constant * light_speed)

    # "Convolution" with incident flux. Data now in units of Watts / steradian.
    # The convolution is just a product with beam_flux * 1s because incident pulse is delta.
    data *= beam_flux 

    # Multiply by detector solid angle (steradians). Data now in Watts.
    # This assumes the detector solid angle is so small, the flux is approx constant over it.
    data *= geometry["detector_solid_angle"]
    
    # Multiply by photon count at given wavelength. Converts data from Watts to photons / s.
    data *= photons_per_joule

    # Switch time units to nanoseconds
    data /= 1e9 # photons per nanosecond
    delta *= 1e9 # Nanoseconds
    
    return data, delta


def get_X(data, metadata):
    return np.linspace(metadata["tmin"], 3*metadata["tmax"], len(data)) / light_speed * 1e9



class Distribution:
    def __init__(self, intensity, delta_t, tmin, noise):
        self.intensity = intensity
        self.delta = delta_t
        self.tmin = tmin
        self.tmax = tmin + intensity.size * delta_t
        self.noise = noise
        self.L = delta_t * sum(intensity)
        self.cdfC = 1 - exp(-noise * tmin)
        self.precompute_I()
        self.precompute_E()
    
    def precompute_I(self):
        N = self.intensity.size
        self.I = np.zeros(N+1)
        for n in range(1, N+1):
            self.I[n] = self.I[n-1] + self.delta * self.intensity[n-1]
    
    def precompute_E(self):
        N = self.intensity.size
        self.E = np.zeros(N+1)
        self.E[0] = 1 - exp(-self.noise * self.tmin)
        for k in range(1, N+1):
            tk0 = self.tmin + (k-1)*self.delta
            tk1 = self.tmin + k*self.delta
            Dk = self.noise + self.intensity[k-1]
            Bk = -self.I[k-1] + self.intensity[k-1] * tk0
            integral = exp(Bk-Dk*tk0) - exp(Bk-Dk*tk1)
            self.E[k] = self.E[k-1] + integral 
    
    def intensity_value(self, t):
        if t < 0 or t >= self.delta * len(self.intensity):
            return 0.0
        i = int(t / self.delta)
        return self.intensity[i]
    
    def integrate_lambda(self, t):
        if t < self.tmin:
            return self.noise * t
        if t > self.tmax:
            return self.noise * t + self.L
        k = int((t - self.tmin) / self.delta)
        tk = self.tmin + k * self.delta
        return self.noise * t + self.I[k] + self.intensity[k] * (t - tk)

    def cdf(self, t):
        if t < self.tmin:
            return 1 - exp(-self.noise * t)
        if t > self.tmax:
            return self.E[-1] + (exp(-self.noise * self.tmax) - exp(-self.noise * t))
    
        k = int((t - self.tmin) / self.delta)
        tk = self.tmin + k*self.delta
    
        D = self.noise + self.intensity[k]
        Ck = exp(-self.I[k] + self.intensity[k] * tk) 
        return self.E[k] + Ck * (exp(-D*tk) - exp(-D*t)) 

    def sample_one(self):
        U = np.random.rand()
        N = len(self.E)
        k = -100
        if U < self.E[0]:
            k = -1
        elif U > self.E[-1]:
            k = N-1
        else:
            for i in range(N-1):
                if U > self.E[i] and U < self.E[i+1]:
                    k = i
                    break
        if k == -100:
            raise ValueError("CDF error: U = {}, E = {}".format(U, self.E))
        if k == N-1:
            return -np.log(self.E[-1] + exp(-self.noise * self.tmax) - U) / self.noise
        if k == -1:
            return -np.log(1 - U) / self.noise
        tk = self.tmin + k*self.delta
        Dk = self.noise + self.intensity[k]
        Bk = -self.I[k] + self.intensity[k] * tk
        Ek = self.E[k]
        return -np.log(Ek + exp(Bk-Dk*tk) - U) / Dk + Bk/Dk
    
    def sample(self, N=1, limit=None):
        S = np.array([self.sample_one() for i in range(N)])
        if not limit is None:
            return np.ma.masked_greater(S, limit)
        else:
            return np.ma.masked_array(S)

def test_create_distribution():
    Y = np.array([1.5, 1.0, 0.5])
    delta = 1.0
    D = Distribution(Y, delta, 0.0, 0.0)
    assert D.tmin == 0.0
    assert D.tmax == 3.0
    assert D.L == 3.0


def test_cdf():
    Y = np.array([1.0])
    delta = 1.0
    D = Distribution(Y, delta, 0.0, 0.1)
    print(D.I)
    print(D.E)
    
    assert D.L == 1.0
    assert D.cdf(0.0) == 0.0

    



if __name__=="__main__":
    from matplotlib import pyplot as plt
    Y = np.array([ 1.0])
    delta = 1.0
    noise = 0.01
    D = Distribution(Y, delta, 1.0, noise)
    
    M = 1000
    S = D.sample(1000, 3.0)
    plt.figure()
    plt.scatter(range(M), S, 1)
    
    plt.figure()
    
    N = 1000
    X = np.linspace(0, 5, N)
    F = np.array([D.cdf(x) for x in X])
    
    theory = np.zeros(N)
    theory[X<1] = 1 - exp(-(Y[0] + noise)*X[X<1])
    theory[X>=1] = 1 - exp(-(Y[0] + noise)) + exp(-noise) - exp(-noise*X[X>=1])
    
    plt.plot(X,F)
    #plt.plot(X,theory)
    plt.show()
