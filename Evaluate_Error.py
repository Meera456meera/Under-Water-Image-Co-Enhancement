import numpy as np
import math

from scipy.ndimage import gaussian_filter


def evaluat_error(sp, act) -> object:
    sigma = 1.5
    mu_actual = gaussian_filter(act, sigma=sigma)
    mu_predicted = gaussian_filter(sp, sigma=sigma)
    sigma_actual_sq = gaussian_filter(act**2, sigma=sigma) - mu_actual**2
    sigma_predicted_sq = gaussian_filter(sp**2, sigma=sigma) - mu_predicted**2
    sigma_actual_predicted = gaussian_filter(act * sp, sigma=sigma) - mu_actual * mu_predicted
    k1 = 0.01
    k2 = 0.03
    max_value = 255
    r = np.squeeze(act)
    x = np.squeeze(sp)
    points = np.zeros(len(x))
    abs_r = np.zeros(len(x))
    abs_x = np.zeros(len(x))
    abs_r_x = np.zeros(len(x))
    abs_x_r = np.zeros(len(x))
    abs_r_x__r = np.zeros(len(x))
    for j in range(1, len(x)):
        points[j] = abs(x[j] - x[j-1])
    for i in range(len(r)):
        abs_r[i] = abs(r[i])
    for i in range(len(r)):
        abs_x[i] = abs(x[i])
    for i in range(len(r)):
        abs_r_x[i] = abs(r[i] - x[i])
    for i in range(len(r)):
        abs_x_r[i] = abs(x[i] - r[i])
    for i in range(len(r)):
        abs_r_x__r[i] = abs((r[i] - x[i]) / r[i])
    md = (100/len(x)) * sum(abs_r_x__r)
    smape = (1/len(x)) * sum(abs_r_x/((abs_r + abs_x) / 2))
    mase = sum(abs_r_x)/((1 / (len(x) - 1)) * sum(points))
    mae = sum(abs_r_x) / len(r)
    rmse = (sum(abs_x_r ** 2) / len(r)) ** 0.5
    onenorm = sum(abs_r_x)
    twonorm = (sum(abs_r_x ** 2) ** 0.5)
    infinitynorm = max(abs_r_x)
    MSE = np.square(np.subtract(sp, act)).mean()
    psnr_value = 10 * math.log10((max_value ** 2) / MSE)
   # SSIM components
    numerator = (2 * mu_actual * mu_predicted + k1) * (2 * sigma_actual_predicted + k2)
    denominator = (mu_actual**2 + mu_predicted**2 + k1) * (sigma_actual_sq + sigma_predicted_sq + k2)

    # Calculate SSIM for each pixel
    ssim_map = numerator / denominator

    # Average SSIM over the image
    ssim_index = np.mean(ssim_map)
    EVAL_ERR = [MSE,psnr_value,ssim_index]
    return EVAL_ERR


