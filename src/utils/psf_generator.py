import numpy as np
import numpy.fft as F
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from .read_mrc import read_mrc
import math


class parameters3D(object):
    # set the parameters3D for 3D SIM
    def __init__(self):
        self.lamda = 488e-3
        self.Nx = 128
        self.Ny = 128
        self.Nz = 11
        self.dx = 61e-3 / 2
        self.dy = 61e-3 / 2
        self.dz = 160e-3
        self.dxy = self.dx
        self.dkx = 1 / (self.Nx * self.dx)
        self.dky = 1 / (self.Ny * self.dy)
        self.dkz = 1 / (self.Nz * self.dz)
        self.dkr = np.min([self.dkx, self.dky])
        self.nphases = 5
        self.ndirs = 3
        self.NA = 1.1
        self.NAEx = 0.5
        self.nimm = 1.406
        self.space = self.lamda / self.NAEx / 2
        self.k0mod = 1 / self.space
        self.norders = int((self.nphases + 1) / 2)


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def cal_psf_3d(otf_path, Ny, Nx, Nz, dky, dkx, dkz):
    headerotf, rawOTF = read_mrc(otf_path)
    dkr = np.min([dkx, dky])
    nxotf = headerotf[0][0]
    nyotf = headerotf[0][1]
    nzotf = headerotf[0][2]
    dkzotf = headerotf[0][10]
    dkrotf = headerotf[0][11]
    lenotf = len(rawOTF)
    rOTF = rawOTF[np.arange(0, lenotf, 2)]
    iOTF = rawOTF[np.arange(1, lenotf, 2)]
    rawOTF = rOTF + 1j * iOTF
    rawOTF = np.abs(np.reshape(rawOTF, (nzotf, nyotf, nxotf)))
    rawOTF = np.transpose(rawOTF, (2, 1, 0))
    diagdist = math.ceil(np.sqrt(np.square(Nx / 2) + np.square(Ny / 2)) + 1)
    z = np.arange(0, nxotf * dkzotf, dkzotf)
    zi = np.arange(0, nxotf * dkzotf, dkz)
    zi = zi[0:-1]
    x = np.arange(0, nyotf * dkrotf, dkrotf)
    xi = np.arange(0, nyotf * dkrotf, dkr)
    xi = xi[0:-1]
    [X, Z] = np.meshgrid(x, z)
    [Xi, Zi] = np.meshgrid(xi, zi)
    rawOTF = rawOTF[:, :, 0]

    OTF1 = []
    for j in range(nxotf):
        curRow = rawOTF[j, :]
        interp = interp1d(x, curRow, 'slinear')
        OTF1.append(interp(xi))
    OTF1 = np.transpose(OTF1, (1, 0))

    OTF2 = []
    for k in range(np.size(OTF1, 0)):
        curCol = OTF1[k]
        interp = interp1d(z, curCol, 'slinear')
        OTF2.append(interp(zi))
    OTF2 = np.transpose(OTF2, (1, 0))

    OTF = F.fftshift(OTF2, 0)
    otfz = np.size(OTF, 0)
    halfotfz = math.ceil(otfz / 2)
    otflen = np.size(OTF, 1)

    prol_OTF = np.zeros((Nz, diagdist))
    prol_OTF[:, 0: otflen] = OTF
    OTF = prol_OTF

    x = np.arange(-Nx / 2, Nx / 2, 1) * dkx
    y = np.arange(-Ny / 2, Ny / 2, 1) * dky
    [X, Y] = np.meshgrid(x, y)
    rdist = np.sqrt(np.square(X) + np.square(Y))

    curOTF = np.zeros((Ny, Nx, Nz))
    otflen = np.size(OTF, 1)
    x = np.arange(0, otflen * dkr, dkr)
    for z in range(Nz):
        OTFz = OTF[z, :]
        interp = interp1d(x, OTFz, 'slinear')
        curOTF[:, :, z] = interp(rdist)

    curOTF = np.abs(curOTF)
    curOTF = curOTF / np.max(curOTF)

    temp = np.zeros_like(curOTF) + 1j * np.zeros_like(curOTF)
    for j in range(Nz):
        temp[:, :, j] = F.fftshift(F.fft2(np.squeeze(curOTF[:, :, j])))
    PSF = np.abs(F.fftshift(F.fft(temp, axis=2), axes=2))
    PSF = PSF / np.sum(PSF)
    return PSF, curOTF


def gaussian_1d(x, *param):
    return param[0] * np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))


def psf_estimator_3d(psf):
    shape = psf.shape
    max_index = np.where(psf == psf.max())
    index_y = max_index[0][0]
    index_x = max_index[1][0]
    index_z = max_index[2][0]
    # estimate y sigma
    x = ar(range(shape[0]))
    y = prctile_norm(np.squeeze(psf[:, index_x, index_z]))
    fit_y, cov_y = curve_fit(gaussian_1d, x, y, p0=[1, index_y, 2])
    print('estimated psf sigma_y: ', fit_y[2])
    # estimate x sigma
    x = ar(range(shape[1]))
    y = prctile_norm(np.squeeze(psf[index_y, :, index_z]))
    fit_x, cov_x = curve_fit(gaussian_1d, x, y, p0=[1, index_x, 2])
    print('estimated psf sigma_x: ', fit_x[2])
    # estimate z sigma
    x = ar(range(shape[2]))
    y = prctile_norm(np.squeeze(psf[index_y, index_x, :]))
    fit_z, cov_z = curve_fit(gaussian_1d, x, y, p0=[1, index_z, 2])
    print('estimated psf sigma_z: ', fit_z[2])
    return fit_y[2], fit_x[2], fit_z[2]






