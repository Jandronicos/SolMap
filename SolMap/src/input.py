from astropy.io import fits
import numpy
import math


# The input class used to store initial values the user provides and the open the fits file being worked on

class ReceiveInput():

    fits_name: str = ""
    ref_long: float = 0
    ref_lat: float = 0
    dx: float = 0
    dy: float = 0
    target_x: int = 0
    target_y: int = 0
    y_centre_out: float = 0
    x_centre_out: float = 0
    hdu_list: fits.HDUList
    fits_header0: fits.Header
    fits_header1:  fits.Header
    fits_data1: numpy.ndarray

    # Opens the first header of a fits file and returns a header object
    def open_header0(cls, hdu_list):
        cls.fits_header0 = hdu_list[0].header
        print(cls.fits_header0.tostring("\n"))

    # Opens the second header of a fits file and returns a header object
    def open_header1(cls, hdu_list):
        cls.fits_header1 = hdu_list[1].header
        print(cls.fits_header1.tostring("\n"))

    # Opens the first header of a fits file and returns a ndarray object
    def open_data0(cls, hdu_list):
        cls.fits_data0 = hdu_list[0].data
        print(cls.fits_data0.shape)

    # Opens the second header of a fits file and returns a ndarray object
    def open_data1(cls, hdu_list):
        fits_hdu1 = hdu_list[1]
        cls.fits_data1 = fits_hdu1.data


    @classmethod
    def get_inputs(cls) -> 'ReceiveInput':
        cls.fits_name = str(input("Fits file location and relative path: "))  # The fits file to be worked on with a relative path
        cls.hdu_list = fits.open(cls.fits_name)
        fits.info(filename=cls.fits_name)
        cls.fits_header0 = cls.open_header0(cls, cls.hdu_list)
        cls.open_header0(cls, cls.hdu_list)
        cls.open_header1(cls, cls.hdu_list)
        cls.open_data1(cls, cls.hdu_list)

        #-------------------------------------------------------------------------------------------------------

        cls.ref_long = float(input("Provide the reference longitude: "))  # The reference longitude
        cls.ref_lat = float(input("Provide the reference latitude: "))  # The reference latitude
        cls.target_x = int(input("Provide the target size x value: "))  # The x size of the output expected, tested on square images
        cls.target_y = int(input("Provide the target size y value: "))  # The y size of the output expected, tested on square images
        cls.y_centre_out = float(cls.target_y / 2)  # Get's the centre of a square image size
        cls.x_centre_out = float(cls.target_x / 2)  # Get's the centre of a square image size
        cls.dx = float(input("Provide the dx value: "))
        cls.dy = float(input("Provide the dy value: "))

        return cls()

