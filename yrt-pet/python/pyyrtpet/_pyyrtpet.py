"""Python interface to YRT-PET code"""

import numpy as np

from . import pyyrtpet as yrt

# %% Raw-data file interface

MAGIC_NUMBER = 732174000


class DataFileRawd:
    """Raw data file with dimension header

    Examples
    --------
    >>> fname = 'tmp.rwd'
    >>> res_check_list = []
    >>> for ndim in range(1, 6):
    ...     dims = np.random.randint(1, 5)
    ...     dat = np.random.random(dims).astype(np.float32)
    ...     DataFileRawd().save(dat, fname)
    ...     dat_check = DataFileRawd().load(fname)
    ...     DataFileRawd().delete(fname)
    ...     res_check_list.append(np.allclose(dat, dat_check))
    >>> np.all(res_check_list)
    True
    """

    def load(self, fname, dtype=np.float32):
        """Load raw data file with dimensions

        Parameters
        ----------
        fname : str
            Name of file to read.
        dtype : numpy.dtype
            Output file type.

        Returns
        -------
        numpy.ndarray
            The parsed array.

        Raises
        ------
        IOError
            If the magic number was not found at the beginning of the file.
        """
        with open(fname, 'r') as fid:
            magic, ndims = np.fromfile(fid, dtype=np.int32, count=2)
            if magic != MAGIC_NUMBER:
                raise IOError('File is not in rawd format.')
            dims = np.fromfile(fid, dtype=np.uint64, count=ndims).astype(int)
            return np.fromfile(fid, dtype=dtype).reshape(dims)

    def save(self, data, fname, **kwargs):
        """Save raw data file with dimensions

        Parameters
        ----------
        data : numpy.ndarray
            Input array.
        fname : str
            Name of output file.
        kwargs : dict
            Ignored.

        See also
        --------
        load_rawd : Load data saved with this function.
        """
        with open(fname, 'wb') as fid:
            head = np.hstack([MAGIC_NUMBER, data.ndim]).astype(np.int32)
            dims = np.array(data.shape).astype(np.uint64)
            head.tofile(fid)
            dims.tofile(fid)
            data.tofile(fid)


# Wrapper for Projection operator
class ProjectionOper:
    def __init__(self, scanner: yrt.Scanner, img_params: yrt.ImageParams,
                 projData: yrt.ProjectionData, projector='Siddon',
                 idx_subset=0, num_subsets=1,
                 tof_width_ps=None, tof_n_std=0,
                 proj_psf_fname=None, num_rays=1):
        self._scanner = scanner
        self._img_params = img_params
        self._projData = projData
        self._idx_subset = idx_subset
        self._num_subsets = num_subsets
        self._binIter = self._projData.getBinIter(self._num_subsets,
                                                  self._idx_subset)
        proj_f = getattr(yrt, 'OperatorProjector{}'.format(projector))
        self._proj_params = yrt.OperatorProjectorParams(
            self._binIter, self._scanner,
            tof_width_ps or np.float32(0), tof_n_std or np.int32(0),
            proj_psf_fname or '', num_rays)
        self._oper = proj_f(self._proj_params)

        self._x = np.require(np.zeros(
            [self._img_params.nz, self._img_params.ny, self._img_params.nx],
            dtype=np.float32), requirements=['C_CONTIGUOUS'])
        self._y = np.require(np.zeros(self._projData.count(),
                                      dtype=np.float32),
                             requirements=['C_CONTIGUOUS'])

    def A(self, x):
        """Forward projection"""
        xx = np.require(x, dtype=np.float32)
        if xx.ndim == 2:
            xx = xx[None, ...]
        img = yrt.ImageAlias(self._img_params)
        img.bind(xx)
        self._y[:] = 0
        projlist = yrt.ProjectionListAlias(self._projData)
        projlist.bind(self._y)
        self._oper.applyA(img, projlist)
        return self._y

    def At(self, y):
        """Backprojection"""
        yy = np.require(y, dtype=np.float32)
        projlist = yrt.ProjectionListAlias(self._projData)
        projlist.bind(yy)
        self._x[:] = 0
        img = yrt.ImageAlias(self._img_params)
        img.bind(self._x)
        self._oper.applyAH(projlist, img)
        return self._x
