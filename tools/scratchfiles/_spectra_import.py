from acoular import PowerSpectra
from acoular.internal import digest
from traits.api import Enum,Trait,Int, Float, CArray, Property, cached_property, property_depends_on
from numpy import fft, arange

class PowerSpectraImport( PowerSpectra ):
    """Provides the cross spectral matrix of multichannel time data
     and its eigen-decomposition.
    
    This class includes the efficient calculation of the full cross spectral
    matrix using the Welch method with windows and overlap. It also contains 
    the CSM's eigenvalues and eigenvectors and additional properties. 
    
    The result is computed only when needed, that is when the :attr:`csm`,
    :attr:`eva`, or :attr:`eve` attributes are acturally read.
    Any change in the input data or parameters leads to a new calculation, 
    again triggered when an attribute is read. The result may be 
    cached on disk in HDF5 files and need not to be recomputed during
    subsequent program runs with identical input data and parameters. The
    input data is taken to be identical if the source has identical parameters
    and the same file name in case of that the data is read from a file.
    """

    #: Sampling frequency of the signal, defaults to 1.0
    sample_freq = Float(1.0, 
        desc="sampling frequency")

    #: Number of samples 
    numchannels = Property(depends_on=['digest'])

    #: 2-element array with the lowest and highest frequency. If set, 
    #: will overwrite :attr:`_freqlc` and :attr:`_freqhc` according to
    #: the range. 
    #: The freq_range interval will be the smallest discrete frequency
    #: inside the half-open interval [_freqlc, _freqhc[ and the smallest
    #: upper frequency outside of the interval.
    #: If user chooses the higher frequency larger than the max frequency,
    #: the max frequency will be the upper bound.
    freq_range = Property(
        desc = "frequency range" )
        
    #: Array with a sequence of indices for all frequencies 
    #: between :attr:`ind_low` and :attr:`ind_high` within the result, readonly.
    indices = Property(
        desc = "index range" )
        
    #: The cross spectral matrix, 
    #: (number of frequencies, numchannels, numchannels) array of complex;
    #: readonly.
    csm = Property( 
        desc="cross spectral matrix")

    # csm shadow trait
    _csm = CArray(
        )

    _csmsum = Float() #: only to trigger digest calculation

    # internal identifier
    digest = Property( 
        depends_on = ['_csmsum', 'sample_freq',
            ], 
        )

    #: Name of the cache file without extension, readonly.
    basename = Property( depends_on = 'digest', 
        desc="basename for cache file")

    block_size = Property() # overwrite mapping trait

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def _get_numchannels( self ):
        return self.csm.shape[1]

    def _get_block_size( self ):
        return (self.csm.shape[0]-1)*2

    def fftfreq ( self ):
        """
        Return the Discrete Fourier Transform sample frequencies.
        
        Returns
        -------
        f : ndarray
            Array of length *block_size/2+1* containing the sample frequencies.
        """
        return abs(fft.fftfreq(self.block_size, 1./self.sample_freq)\
                    [:int(self.block_size/2+1)])


    def _get_csm ( self ):
        return self._csm

    def _set_csm (self, csm):
        self._csmsum = csm.sum() # to trigger new digest creation
        self._csm = csm

    def _get_basename( self ):
        return "csm_import_"+self.digest

    @property_depends_on( 'block_size, ind_low, ind_high' )
    def _get_indices ( self ):
        try:
            return arange(self.block_size/2+1,dtype=int)[ self.ind_low: self.ind_high ]
        except IndexError:
            return range(0)

    #: The :class:`~acoular.tprocess.SamplesGenerator` object that provides the data.
    time_data = Enum(None, 
        desc="import has no time data object")

    calib = Enum(None) # no calib capabilities in this class

    window = Enum(None) # no windowing possible

    overlap = Enum(None) # no overlap possible

    cached = Enum(None)  

    num_blocks = Enum(None)





if __name__ == "__main__":

    ps = PowerSpectraImport()