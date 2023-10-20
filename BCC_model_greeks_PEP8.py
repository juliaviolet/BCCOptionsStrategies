
import numpy as np
import pandas as pd
from scipy.integrate import quad

# defining the required functions for the M76, H93, BCC model
# valuation of European Call and Put Options
# under Stochastic Volatility and Jumps
# 09_gmm / BCC_option_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python

# constants and global variables
# bump used for theta (time decay) calculation
time_bump = 1 / 365.25

# current risk - free rate (obtained from external data source)
# https: / /www.marketwatch.com / investing / bond / tmbmkde - 10y / download - data?startDate = 05 / 14 / 2019&endDate = 05 / 14 / 2020&countryCode = BX
r0 = -0.00067

# characteristic function for the Cox - Ingersoll - Ross CIR) interest rate model


def CIR_char_func_nojit(u, T, kappa, theta, sigma_r, r0):
    gamma = np.sqrt(kappa * *2 + 2 * sigma_r * *2 * u*1j)
    A = 2 * gamma * np.exp((kappa + gamma) * T/2) / \
        (2 * gamma + (gamma + kappa) * (np.exp(gamma * T) - 1))
    B = 2 * (np.exp(gamma * T) - 1) / (2 * gamma + (gamma + kappa) * (np.exp(
gamma * T) - 1))
    char_func_value = np.exp(A * theta - B*r0)
    return char_func_value

# characteristic function for the Heston '93 (H93) stochastic volatility model


def H93_char_func_nojit(u, T, r0, kappa_v, theta_v, sigma_v, rho, v0):
    c1 = kappa_v * theta_v
    c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v) **
                  2 - sigma_v ** 2 * ( - u * 1j - u ** 2))
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) / \
        (kappa_v - rho * sigma_v * u * 1j - c2)
    H1 = (r0 * u * 1j * T + (c1 / sigma_v ** 2) * ((kappa_v - rho * sigma_v *
          u * 1j + c2) * T - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))))
    H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v **
          2 * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value


# characteristic function for the Merton '76 (M76) jump - diffusion model
def M76_char_func_nojit(u, T, lamb, mu, delta):
    omega = -lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    char_func_value = np.exp(
        (1j * u * omega + lamb * (np.exp(
1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1)) * T)
    return char_func_value

# combines the CIR, H93, and M76 models to form the BCC characteristic function


def BCC_char_func_nojit(u, T, r0, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu,
delta, kappa, theta, sigma_r, muV, deltaV, rhoJ):
    CIR1 = CIR_char_func_nojit(u, T, kappa, theta, sigma_r, r0)
    BCC1 = H93_char_func_nojit(u, T, r0, kappa_v, theta_v, sigma_v, rho, v0)
    BCC2 = M76_char_func_nojit(u, T, lamb, mu, delta)
    jumpVolatilityPart = lamb * (np.exp(1j * u * muV * deltaV) - 1) * T
    return CIR1 * BCC1 * BCC2 * np.exp(jumpVolatilityPart * rhoJ)

# integral function used in the BCC model for option valuation


def BCC_int_func_nojit(u, S0, K, T, r0, kappa_v, theta_v, sigma_v, rho, v0,
lamb, mu, delta, kappa, theta, sigma_r, muV, deltaV, rhoJ):
    char_func_value = BCC_char_func_nojit(u - 1j * 0.5, T, r0, kappa_v, theta_v,

                                          sigma_v, rho, v0, lamb, mu, delta,
kappa, theta, sigma_r, muV, deltaV, rhoJ)
    int_func_value = 1 / (u ** 2 + 0.25) * \
        (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value


def BCC_option_value_nojit(S0, K, T, r0, kappa_v, theta_v, sigma_v, rho, v0,
lamb, mu, delta, kappa, theta, sigma_r, muV, deltaV, rhoJ, option_type):
     # Calculate option value (Call or Put) using the BCC model.
    int_value = quad(lambda u: BCC_int_func_nojit(u, S0, K, T, r0, kappa_v,
theta_v, sigma_v, rho,
                     v0, lamb, mu, delta, kappa, theta, sigma_r, muV, deltaV,
rhoJ), 0, np.inf, limit = 250)[0]

    if option_type == 'CALL':
        option_value = max(0, S0 - np.exp( - r0 * T) *
                           np.sqrt(S0 * K) / np.pi * int_value)
    else:   # 'PUT'
        call_value = max(0, S0 - np.exp( - r0 * T) *
                         np.sqrt(S0 * K) / np.pi * int_value)
        option_value = call_value + K * np.exp( - r0 * T) - S0

    return option_value

# Delta


def BCC_delta_nojit(S0, K, T, r0, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu,
delta, kappa, theta, sigma_r, muV, deltaV, rhoJ, price_change_abs,
option_type):
    epsilon = price_change_abs if price_change_abs != 0 else 0.01
    price_plus = BCC_option_value_nojit(S0 + epsilon, K, T, r0, kappa_v,
theta_v, sigma_v,
                                        rho, v0, lamb, mu, delta, kappa, theta,
sigma_r, muV, deltaV, rhoJ, option_type)
    price_minus = BCC_option_value_nojit(S0 - epsilon, K, T, r0, kappa_v,
theta_v, sigma_v,
                                         rho, v0, lamb, mu, delta, kappa, theta,
sigma_r, muV, deltaV, rhoJ, option_type)
    delta = (price_plus - price_minus) / (2 * epsilon)
    return delta

    if option_type == 'CALL':
        delta = (price_plus - price_minus) / (2 * epsilon)
    elif option_type == 'PUT ':
        delta = (price_minus - price_plus) / (2 * epsilon)
    else:
        delta = 0   # This should not happen with the provided data,
but added as a safeguard

    return delta

# Vega


def BCC_vega_nojit(S0, K, T, r0, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu,
delta, kappa, theta, sigma_r, muV, deltaV, rhoJ, price_change_abs,
option_type):
    epsilon = 0.01
    vega_plus = BCC_option_value_nojit(S0, K, T, r0, kappa_v, theta_v,
sigma_v + epsilon,
                                       rho, v0, lamb, mu, delta, kappa, theta,
sigma_r, muV, deltaV, rhoJ, option_type)
    vega_minus = BCC_option_value_nojit(S0, K, T, r0, kappa_v, theta_v,
sigma_v - epsilon,
                                        rho, v0, lamb, mu, delta, kappa, theta,
sigma_r, muV, deltaV, rhoJ, option_type)
    vega = (vega_plus - vega_minus) / (2 * epsilon)
    return vega

# Theta


def BCC_theta_nojit(S0, K, T, r0, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu,
delta, kappa, theta, sigma_r, muV, deltaV, rhoJ, option_type):
    theta_value = BCC_option_value_nojit(S0, K, T + time_bump, r0, kappa_v,
theta_v, sigma_v,
                                         rho, v0, lamb, mu, delta, kappa, theta,
sigma_r, muV, deltaV, rhoJ, option_type)
    theta_value_bumped = BCC_option_value_nojit(S0, K, T - time_bump, r0,
kappa_v, theta_v,
                                                sigma_v, rho, v0, lamb, mu,
delta, kappa, theta, sigma_r, muV, deltaV, rhoJ, option_type)
    theta = (theta_value_bumped - theta_value) / (2 * time_bump)
    return theta

# Gamma


def BCC_gamma_nojit(S0, K, T, r0, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu,
delta, kappa, theta, sigma_r, muV, deltaV, rhoJ, price_change_abs,
option_type):
    epsilon = price_change_abs if price_change_abs != 0 else 0.01
    delta_current = BCC_delta_nojit(S0, K, T, r0, kappa_v, theta_v, sigma_v,
rho, v0, lamb,
                                    mu, delta, kappa, theta, sigma_r, muV,
deltaV, rhoJ, price_change_abs, option_type)
    delta_plus = BCC_delta_nojit(S0 + epsilon, K, T, r0, kappa_v, theta_v,
sigma_v, rho, v0,
                                 lamb, mu, delta, kappa, theta, sigma_r, muV,
deltaV, rhoJ, price_change_abs, option_type)
    delta_minus = BCC_delta_nojit(S0 - epsilon, K, T, r0, kappa_v, theta_v,
sigma_v, rho, v0,
                                  lamb, mu, delta, kappa, theta, sigma_r, muV,
deltaV, rhoJ, price_change_abs, option_type)
    gamma = (delta_plus - 2 * delta_current + delta_minus) / (epsilon ** 2)
    return abs(gamma)

    if option_type == 'CALL':
        gamma = (delta_plus - 2 * delta_current + delta_minus) / (epsilon ** 2)
    elif option_type == 'PUT ':
        gamma = (delta_minus - 2 * delta_current + delta_plus) / (epsilon ** 2)
    else:
        gamma = 0   # This should not happen with the provided data,
but added as a safeguard

    return abs(gamma)
