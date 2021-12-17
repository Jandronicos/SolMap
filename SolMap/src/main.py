# SolMap Project V2
# Created by Jacob Andronicos
# SolMap uses AstroPy to read, modiffy and create fits files
# SolMap uses Numba and Numpy for just in time compliation in C for faster run times.
# SolMap performs a postel projection based on user inputs and outs a new fits file
# with the modified data and header values


import numpy
import input
import interpolation
import matplotlib.pyplot as plt
import time
import numba


@numba.jit(nopython=True)
def postel_projection(b0:float, l0:float, crota2:float,
                      rsun_ref:float, dsun_obs:float,
                      rsun_obs:float, cdelt1:float,
                      nyCL:float, nxCL:float,
                      crpix1:float, crpix2:float,
                      target_y:float, y_centre_out:float,
                      target_x:float, x_centre_out:float,
                      lon_ref:float, lat_ref:float,
                      fits_data1: numba.types.float64[:,:], PI:float,
                      rad_fact:float, maj_ax_ang:float,
                      maj_ax_fact:float, min_ax_fact:float,
                      dy: float, dx: float):

    buff_out: numpy.ndarray = numpy.empty((int(target_x), int(target_y)), dtype='float')
    buff_in: numpy.ndarray = fits_data1

    # Preliminary value definitions
    b0rad = b0*rad_fact
    p_angle_rad = crota2 * rad_fact
    lon_rad = (lon_ref-l0)*rad_fact
    lat_rad = lat_ref*rad_fact
    psi = maj_ax_ang*rad_fact
    sun_rad_arcsec = numpy.arcsin(rsun_ref / dsun_obs) * 3600.0 / (2.0 * PI) * 360.0
    sin_alpha_max = numpy.sin(sun_rad_arcsec*PI/(180.0*3600.0))
    sun_rad_pix = rsun_obs / cdelt1
    rad_maj = maj_ax_fact * sun_rad_pix;
    rad_min = min_ax_fact * sun_rad_pix;

    cos_b0 = numpy.cos(-b0rad)
    sin_b0 = numpy.sin(-b0rad)
    cos_p = numpy.cos(-p_angle_rad)
    sin_p = numpy.sin(-p_angle_rad)
    cos_lon = numpy.cos(-lon_rad)
    sin_lon = numpy.sin(-lon_rad)
    cos_lat = numpy.cos(lat_rad)
    sin_lat = numpy.sin(lat_rad)
    cos_2psi = numpy.cos(2. * psi)
    sin_2psi = numpy.sin(2. * psi)

    # Calculations
    rot_xx = cos_p * cos_lon + sin_p * sin_b0 * sin_lon
    rot_xy = cos_p * sin_lon * sin_lat + sin_p * (cos_b0 * cos_lat - sin_b0 * sin_lat * cos_lon)
    rot_xz = -cos_p * sin_lon * cos_lat + sin_p * (cos_b0 * sin_lat + sin_b0 * cos_lat * cos_lon)
    rot_yx = -sin_p * cos_lon + cos_p * sin_b0 * sin_lon
    rot_yy = -sin_p * sin_lon * sin_lat + cos_p * (cos_b0 * cos_lat - sin_b0 * sin_lat * cos_lon)
    rot_yz = sin_p * sin_lon * cos_lat + cos_p * (cos_b0 * sin_lat + sin_b0 * cos_lat * cos_lon)
    rot_zx = cos_b0 * sin_lon
    rot_zy = -sin_b0 * cos_lat - cos_b0 * sin_lat * cos_lon
    rot_zz = -sin_b0 * sin_lat + cos_b0 * cos_lat * cos_lon

    squeez_xx = 0.5 * ((rad_maj + rad_min) - (rad_maj - rad_min) * cos_2psi)
    squeez_xy = 0.5 * (rad_maj - rad_min) * sin_2psi
    squeez_yx = 0.5 * (rad_maj - rad_min) * sin_2psi
    squeez_yy = 0.5 * ((rad_maj + rad_min) + (rad_maj - rad_min) * cos_2psi)

    for i in range(0, int(target_y)):
        eta = dy*(i-y_centre_out)
        eta2 = eta*eta

        for j in range(0, int(target_x)):
            xi = dx*(j-x_centre_out)
            xi2 = xi*xi
            if eta == 0 and xi == 0:
                phi = 0.0
            else:
                phi = numpy.arctan2(eta, xi)
            rho2 = xi2+eta2
            rho = numpy.sqrt(rho2)

            zeta = numpy.cos(rho)
            rho = numpy.sin(rho)
            new_xi = rho*numpy.cos(phi)
            new_eta = rho*numpy.sin(phi)

            z = rot_zx*new_xi + rot_zy*new_eta + rot_zz*zeta

            if z > sin_alpha_max:
                x = rot_xx*new_xi + rot_xy*new_eta + rot_xz*zeta
                y = rot_yx*new_xi + rot_yy*new_eta + rot_yz*zeta

                parallax = numpy.sqrt(1.0 - sin_alpha_max*sin_alpha_max)/(1.0 - sin_alpha_max*z)
                x *= parallax
                y *= parallax

                temp =x
                x = crpix1 + squeez_xx * temp + squeez_xy * y
                y = crpix2 + squeez_yx * temp + squeez_yy * y

                temp_buff_value = interpolation.bi_cubic_interp(x, y, buff_in, float(nyCL), float(nxCL))
                buff_out[j,i] = temp_buff_value
            else:
                buff_out[j,i] = dark_side_value
    return buff_out



if __name__ == '__main__':
    inputs = input.ReceiveInput
    inputs.get_inputs()

    #Values are given explicit values here to comply with numba definitions for compiling in C with strict typing

    b0: float = float(inputs.fits_header1['CRLT_OBS'])
    l0: float = float(inputs.fits_header1['CRLN_OBS'])
    crota2:float = inputs.fits_header1['CROTA2']
    rsun_ref:float = inputs.fits_header1['RSUN_REF']
    dsun_obs:float = inputs.fits_header1['DSUN_OBS']
    rsun_obs:float = inputs.fits_header1['RSUN_OBS']
    cdelt1:float = inputs.fits_header1['CDELT1']
    nyCL:float = inputs.fits_header1['NAXIS2']
    nxCL:float = inputs.fits_header1['NAXIS1']
    crpix1:float = inputs.fits_header1['CRPIX1']
    crpix2:float = inputs.fits_header1['CRPIX2']
    start_time = time.time()
    target_y:float = inputs.target_y
    y_centre_out:float = inputs.y_centre_out
    target_x:float = inputs.target_x
    x_centre_out:float = inputs.x_centre_out
    dy: float = inputs.dy
    dx: float = inputs.dx
    ref_long:float = inputs.ref_long
    ref_lat:float = inputs.ref_lat
    fits_data1:numpy.ndarray = inputs.fits_data1.astype(numpy.float64)
    # Basic numbers used in calcs as global variables
    PI: float = 3.14159265359
    rad_fact: float = PI / 180.0
    maj_ax_ang: float = 0.0
    maj_ax_fact: float = 1.0
    dark_side_value: float = 0.0
    min_ax_fact: float = 1.0
    plt.imshow(numpy.squeeze(fits_data1), origin='lower', cmap='gray')
    plt.show()
    buff_out = postel_projection(b0, l0, crota2, rsun_ref, dsun_obs, rsun_obs,
                                 cdelt1, nyCL, nxCL, crpix1, crpix2, target_y,
                                 y_centre_out, target_x, x_centre_out, ref_long,
                                 ref_lat, fits_data1, PI, rad_fact, maj_ax_ang, maj_ax_fact, min_ax_fact,
                                 dy, dx)
    end_time = time.time()
    total_time = end_time - start_time
    print(
        "The total time taken in seconds was: " + str(total_time) + ". In minutes it is: " + str(total_time / 60) + ".")
    inputs.hdu_list[1].data = buff_out
    plt.imshow(numpy.squeeze(inputs.hdu_list[1].data), origin='lower', cmap='gray')
    plt.show()
    new_name: str = "POSTEL" + inputs.fits_name
    inputs.hdu_list[1].header['CRLN_REF'] = inputs.ref_long
    inputs.hdu_list[1].header['CRLT_REF'] = inputs.ref_lat
    inputs.hdu_list[1].header['X0'] = inputs.x_centre_out
    inputs.hdu_list[1].header['Y0'] = inputs.y_centre_out
    inputs.hdu_list[1].header['DAXIS1'] = inputs.dx
    inputs.hdu_list[1].header['DAXIS2'] = inputs.dy
    inputs.hdu_list.writeto(new_name, output_verify="fix", overwrite=True, checksum=True)

