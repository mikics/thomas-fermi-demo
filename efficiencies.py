from scipy.special import spherical_jn, spherical_yn
import numpy as np

epsilon_0 = 8.8541878128 * 10**-12  # Vacuum permittivity

def spherical_in(l, x, derivative=False):

    j = spherical_jn(l, x, derivative=derivative)
    y = spherical_yn(l, x, derivative=derivative)

    return j + 1j*y

def calculate_analytical_efficiencies(radius_sph, gcs, k0, omega0, gamma, omega_p, beta, _scaling_m, _f_afac, _beta_afac):

    sigma = 1j*epsilon_0*(omega_p / _f_afac)**2/(
        omega0 / _f_afac + 1j*gamma / _f_afac)
    k_d = k0 * _scaling_m
    eps_m = 1 - sigma/(1j*epsilon_0*omega0 / _f_afac)
    eps_d = 1
    eps_inf = 1
    k_m = k0 * _scaling_m * np.sqrt(eps_m)
    k_nl = (omega_p / _f_afac / (beta / _beta_afac)
            ) * np.sqrt(eps_m / (eps_inf * (eps_inf - eps_m)))
    x_d = k_d*radius_sph / _scaling_m
    x_m = k_m*radius_sph / _scaling_m
    x_nl = k_nl*radius_sph / _scaling_m
    
    num_l = 50
    for l in range(1, num_l + 1):
    
        jl_xd = spherical_jn(l, x_d, derivative=False)
        jl_xm = spherical_jn(l, x_m, derivative=False)
        jl_xnl = spherical_jn(l, x_nl, derivative=False)
        jl_xd_p = spherical_jn(l, x_d, derivative=True)
        jl_xm_p = spherical_jn(l, x_m, derivative=True)
        jl_xnl_p = spherical_jn(l, x_nl, derivative=True)
    
        il_xd = spherical_in(l, x_d, derivative=False)
        il_xd_p = spherical_in(l, x_d, derivative=True)
    
        delta_l = l*(l+1)*jl_xm*(eps_m - eps_inf)/eps_inf*jl_xnl/x_nl/jl_xnl_p
    
        tm_num = (-eps_m * jl_xm * (x_d * jl_xd_p + jl_xd) + eps_d * jl_xd *
                  (x_m * jl_xm_p + jl_xm + delta_l))
    
        tm_den = (
            eps_m * jl_xm * (x_d * il_xd_p + il_xd) - eps_d * il_xd *
            (x_m * jl_xm_p + jl_xm + delta_l))
    
        t_TM = tm_num/tm_den
    
        te_num = -jl_xm*(x_d*jl_xd_p+jl_xd) + jl_xd*(x_m*jl_xm_p+jl_xm)
        te_den = jl_xm*(x_d*il_xd_p+il_xd) - il_xd*(x_m*jl_xm_p+jl_xm)
    
        t_TE = te_num/te_den
        if l == 1:
    
            q_ext = 2*np.pi/k_d**2*(2*l+1)*np.real(
                t_TM + t_TE)/(gcs / _scaling_m**2)
            q_sca = 2*np.pi/k_d**2*(2*l+1)*(np.abs(t_TM) **
                                            2 + np.abs(t_TE)**2)/(gcs / _scaling_m**2)
    
        else:
    
            q_ext += 2*np.pi/k_d**2*(2*l+1)*np.real(
                t_TM + t_TE)/(gcs / _scaling_m**2)
            q_sca += 2*np.pi/k_d**2*(2*l+1)*(np.abs(t_TM)
                                             ** 2 + np.abs(t_TE)**2)/(gcs / _scaling_m**2)
    
            q_abs_analyt = -q_ext - q_sca

            return q_abs_analyt
