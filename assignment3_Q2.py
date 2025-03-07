import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Reference temperature in Celsius
REF_TEMP = 60

# Dataset containing frequency (w_Gp, w_Gpp), storage modulus (Gp), and loss modulus (Gpp) at different temperatures
dataset = {
    40: {
        'w_Gp': [246.9134,194.5251,149.7568,112.5724,86.64758,68.27689,51.32383,36.78915,36.78915,27.63796,19.80707,15.61075,12.0109,9.694934,7.118646,4.317568,4.317568,3.402851,2.557928

],
        'Gp': [911580.1, 677241.9, 555532.9, 440896.6, 349915.9, 268690.1, 213244.9, 163744.5, 163744.5, 117700, 87443.09, 69398.88, 51558.64, 42292.89, 34692.32, 21142.11, 21142.11, 16779.35, 13316.87],
        'w_Gpp': [225.5496, 169.4448, 105.379, 105.379, 75.5361, 55.46348, 29.18015, 29.18015, 22.46459, 16.89674, 12.69878, 10.4915, 8.078576, 5.266625, 3.958142, 2.974749, 2.235678, 1.423119, 0.863315, 0.523822, 0.325445, 0.227733, 0.16715, 0.106399],
        'Gpp': [1823534, 1310762, 1006496, 1006496, 772858.2, 633965.8, 373802.1, 373802.1, 306625.1, 268690.1, 206319.2, 158426.4, 134317.5, 99788.74, 76624.82, 58837.94, 45179.91, 32475.46, 20455.46, 13316.87, 8669.522, 6231.679, 4785.123, 3439.561]
    },
    50: {
        'w_Gp': [215.8932, 162.2808, 121.9818, 81.76522, 58.03893, 40.02468, 27.60173, 15.59525, 9.866912, 7.003773, 4.691314],
        'Gp': [176081, 138826.2, 109453.8, 73648.29, 53642.4, 36094.43, 24286.9, 15096.99, 8669.522, 6314.525, 3772.7],
        'w_Gpp': [226.8095, 131.2294, 77.79276, 38.09983, 20.52089, 13.37808, 8.112398, 4.807117, 2.411204, 1.298954, 0.734272, 0.489766, 0.29705, 0.123075, 0.123075],
        'Gpp': [537490.5, 338551.5, 235448.3, 125734.5, 71728.45, 53289.35, 31420.74, 20455.46, 11290.34, 6657.07, 4479.351, 2916.137, 1777.145, 728.7352, 728.7352]
    },
    60: {
        'w_Gp': [194.5564, 116.1541, 72.14853, 44.82213, 15.9655, 11.39644, 7.210284, 4.391889, 2.190478, 1.501377, 1.09215, 0.748571, 0.49307, 0.33108, 0.255477, 0.182152, 0.127252, 0.127252, 0.104231],
        'Gp': [39808.81, 24260.19, 15197.01, 9785.24, 3255.402, 2405.27, 1209.023, 778.4806, 322.756, 196.6932, 126.6492, 77.18225, 44.51783, 23.64303, 14.81041, 9.025725, 4.927202, 4.927202, 3.352061],
        'w_Gpp': [196.2405, 134.6615, 83.68589, 52.00692, 34.97309, 22.60866, 15.82587, 9.636571, 7.154333, 4.272698, 2.761207, 1.857136, 1.109299, 0.776114, 0.463356, 0.324184, 0.178684],
        'Gpp': [166470, 122997.2, 83677.14, 56927.02, 38728.44, 26347.64, 20010.13, 12194.52, 8765.462, 5644.014, 3634.137, 2541.336, 1681.994, 1176.211, 716.8038, 501.2577, 258.9888]
    },
    70: {
        'w_Gp': [171.9269, 130.2268, 98.57549, 74.64171, 49.21397, 30.55888, 20.13188, 12.74968, 7.605511, 5.007943, 3.234211, 2.305192, 1.430198, 0.998806, 0.726564, 0.528264, 0.391736, 0.22453, 0.173459, 0.173459, 0.109798, 0.086585],
        'Gp': [10058.21, 7638.867, 5196.855, 3735.515, 2541.336, 1506.704, 893.2924, 529.6139, 281.2733, 153.5489, 88.56524, 51.08341, 26.39367, 13.63703, 8.780777, 5.205934, 3.086487, 1.509336, 1.146289, 1.146289, 0.625767, 0.545339],
        'w_Gpp': [177.0992, 109.6039, 64.63092, 34.49411, 17.80955, 10.3264, 5.979228, 3.40991, 1.976601, 1.163787, 0.744834, 0.468994, 0.299994, 0.173896, 0.113045, 0.081273],
        'Gpp': [52177.03, 30793.11, 21336.9, 12306.88, 6780.305, 4385.852, 2255.705, 1426.033, 881.0875, 474.4197, 313.9967, 212.6401, 128.4036, 79.33533, 45.7597, 37.22755]
    },
    80: {
        'w_Gp': [162.1619, 94.19524, 25.26832, 14.25642, 9.822102, 6.767033, 4.529495, 2.865752, 1.918183, 1.246786, 0.76637, 0.484641, 0.306919, 0.205533, 0.09805],
        'Gp': [2641.137, 1457.77, 310.902, 152.3705, 87.49952, 50.24703, 26.6564, 15.30757, 8.120772, 3.676725, 1.950528, 1.034769, 0.696266, 0.399834, 0.302993],
        'w_Gpp': [172.703, 77.35098, 46.66337, 36.26272, 24.26852, 18.62926, 12.33071, 6.411065, 4.542106, 3.146565, 1.654129, 1.185483, 0.740149, 0.45185, 0.32377, 0.229495],
        'Gpp': [20148.25, 9999.446, 5912.62, 4962.661, 3496.101, 2542.631, 1909.031, 1076.148, 746.15, 551.3668, 291.6348, 205.4512, 123.432, 79.03273, 53.93197, 40.49261]
    },
    90: {
        'w_Gp': [150.9115, 55.38211, 22.16376, 16.1586, 10.2087, 6.273551, 3.340099, 2.579292, 1.191154, 0.403062, 0.255317, 0.125308, 0.125308, 0.08174],
        'Gp': [804.6125, 201.0706, 63.73107, 35.17622, 15.92623, 8.120772, 3.264681, 2.02936, 0.784144, 0.384303, 0.269037, 0.203875, 0.203875, 0.167236],
        'w_Gpp': [157.4234, 102.5671, 64.95481, 39.98341, 27.55353, 16.02418, 11.36892, 7.620661, 5.108179, 3.625908, 2.167383, 1.453505, 1.031979, 0.774784, 0.476924, 0.338451, 0.247044, 0.18041, 0.104895, 0.070278],
        'Gpp': [8009.06, 5389.066, 3626.147, 2439.93, 1457.77, 980.8912, 660.0134, 444.104, 298.8248, 217.6518, 124.9876, 91.03585, 68.98661, 44.61593, 30.02077, 21.01651, 14.14141, 10.30002, 6.661366, 4.140775]
    }
}

# Extract reference data (60°C)
ref_freq_Gp = dataset[REF_TEMP]['w_Gp']
ref_modulus_Gp = dataset[REF_TEMP]['Gp']
ref_freq_Gpp = dataset[REF_TEMP]['w_Gpp']
ref_modulus_Gpp = dataset[REF_TEMP]['Gpp']

# Select reference points (all points)
ref_modulus_Gp_sub = ref_modulus_Gp
ref_freq_Gp_sub = ref_freq_Gp
ref_modulus_Gpp_sub = ref_modulus_Gpp
ref_freq_Gpp_sub = ref_freq_Gpp

# Compute shift factors (a_T) for each temperature
temp_list = sorted(dataset.keys())
shift_factors_Gp = []
shift_factors_Gpp = []

for temp in temp_list:
    if temp == REF_TEMP:
        shift_factors_Gp.append(1.0)
        shift_factors_Gpp.append(1.0)
        continue
    
    # Compute shift factor for G'
    freq_Gp = np.array(dataset[temp]['w_Gp'])
    modulus_Gp = np.array(dataset[temp]['Gp'])
    log_freq_Gp = np.log(freq_Gp)
    log_modulus_Gp = np.log(modulus_Gp)
    coeffs_Gp = np.polyfit(log_freq_Gp, log_modulus_Gp, 1)
    log_freq_pred_Gp = (np.log(ref_modulus_Gp_sub) - coeffs_Gp[1]) / coeffs_Gp[0]
    shift_factor_Gp = np.exp(np.log(ref_freq_Gp_sub) - log_freq_pred_Gp)
    shift_factors_Gp.append(np.exp(np.mean(np.log(shift_factor_Gp))))
    
    # Compute shift factor for G''
    freq_Gpp = np.array(dataset[temp]['w_Gpp'])
    modulus_Gpp = np.array(dataset[temp]['Gpp'])
    log_freq_Gpp = np.log(freq_Gpp)
    log_modulus_Gpp = np.log(modulus_Gpp)
    coeffs_Gpp = np.polyfit(log_freq_Gpp, log_modulus_Gpp, 1)
    log_freq_pred_Gpp = (np.log(ref_modulus_Gpp_sub) - coeffs_Gpp[1]) / coeffs_Gpp[0]
    shift_factor_Gpp = np.exp(np.log(ref_freq_Gpp_sub) - log_freq_pred_Gpp)
    shift_factors_Gpp.append(np.exp(np.mean(np.log(shift_factor_Gpp))))

# Convert to arrays
temp_list = np.array(temp_list)
shift_factors_Gp = np.array(shift_factors_Gp)
shift_factors_Gpp = np.array(shift_factors_Gpp)

# Plot shift factors for G' and G''
plt.figure(figsize=(8, 5))
plt.semilogy(temp_list, shift_factors_Gp, 's-', color='#1f77b4', label='Shift Factors (G\')', markersize=8, linewidth=2)
plt.semilogy(temp_list, shift_factors_Gpp, 'd-', color='#ff7f0e', label='Shift Factors (G\'\')', markersize=8, linewidth=2)
plt.plot(REF_TEMP, 1, 'o', color='#2ca02c', label=f'Reference ({REF_TEMP}°C)', markersize=10)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Shift Factor (a_T)', fontsize=12)
plt.title('Shift Factors vs Temperature', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Generate TTS Master Curves for G' and G''
def generate_master_curve(modulus_type, shift_factors, ref_temp):
    plt.figure(figsize=(10, 6))
    
    # Collect all shifted data points
    all_shifted_freq = []
    all_modulus = []
    
    for idx, temp in enumerate(temp_list):
        freq = np.array(dataset[temp][f'w_{modulus_type}'])
        modulus = np.array(dataset[temp][modulus_type])
        shifted_freq = freq * shift_factors[idx]
        all_shifted_freq.extend(shifted_freq)
        all_modulus.extend(modulus)
    
    # Convert to arrays and sort
    all_shifted_freq = np.array(all_shifted_freq)
    all_modulus = np.array(all_modulus)
    sort_idx = np.argsort(all_shifted_freq)
    all_shifted_freq_sorted = all_shifted_freq[sort_idx]
    all_modulus_sorted = all_modulus[sort_idx]
    
    # Define sigmoidal (logistic) function for non-linear fit
    def sigmoid(x, A, B, C, D):
        return A / (1 + np.exp(-B * (np.log10(x) - C))) + D
    
    # Fit the sigmoidal model to the data
    params, _ = curve_fit(
        sigmoid,
        all_shifted_freq_sorted,
        np.log10(all_modulus_sorted),
        p0=[10, 1, 0, 4],  # Initial guess for A, B, C, D
        maxfev=10000
    )
    
    # Generate best-fit curve
    x_fit = np.logspace(np.log10(all_shifted_freq_sorted.min()), np.log10(all_shifted_freq_sorted.max()), 500)
    y_fit_log = sigmoid(x_fit, *params)
    y_fit = 10**y_fit_log  # Convert back from log10 space
    
    # Plot shifted data and best-fit curve
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, temp in enumerate(temp_list):
        freq = np.array(dataset[temp][f'w_{modulus_type}'])
        modulus = np.array(dataset[temp][modulus_type])
        shifted_freq = freq * shift_factors[idx]
        if temp == ref_temp:
            plt.loglog(shifted_freq, modulus, 'o', color=colors[idx % len(colors)], markersize=8, label=f'{temp}°C (Reference)', alpha=0.8)
        else:
            plt.loglog(shifted_freq, modulus, 'o', color=colors[idx % len(colors)], markersize=6, label=f'{temp}°C', alpha=0.8)
    
    plt.loglog(x_fit, y_fit, 'b-', linewidth=3, label='Best-Fit Sigmoidal Curve')
    
    plt.xlabel('Shifted Frequency (ω × a_T) [rad/s]', fontsize=12)
    plt.ylabel(f"{modulus_type} [Pa]", fontsize=12)
    plt.title(f'Master Curve for {modulus_type} with Best-Fit Sigmoidal Function', fontsize=14)
    plt.legend(ncol=2, fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.show()
    
    # Print best-fit parameters
    A, B, C, D = params
    print(f"Sigmoidal Fit Parameters for {modulus_type}:")
    print(f"A (Amplitude) = {A:.2f}")
    print(f"B (Slope) = {B:.2f}")
    print(f"C (Midpoint Frequency) = {10**C:.2e} rad/s")
    print(f"D (Lower Plateau) = {10**D:.2e} Pa")

# Generate master curves for G' and G''
generate_master_curve('Gp', shift_factors_Gp, REF_TEMP)
generate_master_curve('Gpp', shift_factors_Gpp, REF_TEMP)

# Fit WLF and Arrhenius equations (exclude reference temperature)
mask = temp_list != REF_TEMP
temp_fit = temp_list[mask]
log_shift_factors_Gp_fit = np.log(shift_factors_Gp[mask])
log_shift_factors_Gpp_fit = np.log(shift_factors_Gpp[mask])

# WLF model
def wlf_model(T, C1, C2):
    return -C1 * (T - REF_TEMP) / (C2 + T - REF_TEMP)

# Arrhenius model
def arrhenius_model(T, E):
    R = 8.314  # J/mol·K
    T_k = T + 273.15
    T_ref_k = REF_TEMP + 273.15
    return (E / R) * (1/T_k - 1/T_ref_k)

# Fit WLF for G'
params_wlf_Gp, _ = curve_fit(wlf_model, temp_fit, log_shift_factors_Gp_fit, p0=[17, 50])
C1_Gp, C2_Gp = params_wlf_Gp

# Fit Arrhenius for G'
params_arr_Gp, _ = curve_fit(arrhenius_model, temp_fit, log_shift_factors_Gp_fit, p0=1e5)
E_Gp = params_arr_Gp[0]

# Fit WLF for G''
params_wlf_Gpp, _ = curve_fit(wlf_model, temp_fit, log_shift_factors_Gpp_fit, p0=[17, 50])
C1_Gpp, C2_Gpp = params_wlf_Gpp

# Fit Arrhenius for G''
params_arr_Gpp, _ = curve_fit(arrhenius_model, temp_fit, log_shift_factors_Gpp_fit, p0=1e5)
E_Gpp = params_arr_Gpp[0]

# Calculate predicted values
log_shift_factors_Gp_wlf = wlf_model(temp_fit, C1_Gp, C2_Gp)
log_shift_factors_Gp_arr = arrhenius_model(temp_fit, E_Gp)
log_shift_factors_Gpp_wlf = wlf_model(temp_fit, C1_Gpp, C2_Gpp)
log_shift_factors_Gpp_arr = arrhenius_model(temp_fit, E_Gpp)

# Calculate R-squared
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

r2_wlf_Gp = r_squared(log_shift_factors_Gp_fit, log_shift_factors_Gp_wlf)
r2_arr_Gp = r_squared(log_shift_factors_Gp_fit, log_shift_factors_Gp_arr)
r2_wlf_Gpp = r_squared(log_shift_factors_Gpp_fit, log_shift_factors_Gpp_wlf)
r2_arr_Gpp = r_squared(log_shift_factors_Gpp_fit, log_shift_factors_Gpp_arr)

# Plot fits for G'
plt.figure(figsize=(8, 5))
plt.plot(temp_fit, log_shift_factors_Gp_fit, 'o', color='#1f77b4', label='Data (G\')', markersize=8)
plt.plot(temp_fit, log_shift_factors_Gp_wlf, '-', color='#ff7f0e', label=f'WLF Fit (R²={r2_wlf_Gp:.3f})', linewidth=2)
plt.plot(temp_fit, log_shift_factors_Gp_arr, '--', color='#2ca02c', label=f'Arrhenius Fit (R²={r2_arr_Gp:.3f})', linewidth=2)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('log(a_T)', fontsize=12)
plt.title('Shift Factor Fits for G\'', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot fits for G''
plt.figure(figsize=(8, 5))
plt.plot(temp_fit, log_shift_factors_Gpp_fit, 'o', color='#1f77b4', label='Data (G\'\')', markersize=8)
plt.plot(temp_fit, log_shift_factors_Gpp_wlf, '-', color='#ff7f0e', label=f'WLF Fit (R²={r2_wlf_Gpp:.3f})', linewidth=2)
plt.plot(temp_fit, log_shift_factors_Gpp_arr, '--', color='#2ca02c', label=f'Arrhenius Fit (R²={r2_arr_Gpp:.3f})', linewidth=2)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('log(a_T)', fontsize=12)
plt.title('Shift Factor Fits for G\'\'', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Determine better model for G'
if r2_wlf_Gp > r2_arr_Gp:
    print("WLF equation provides a better fit for G\'.")
else:
    print("Arrhenius equation provides a better fit for G\'.")

# Determine better model for G''
if r2_wlf_Gpp > r2_arr_Gpp:
    print("WLF equation provides a better fit for G\'\'.")
else:
    print("Arrhenius equation provides a better fit for G\'\'.")

print(f"WLF parameters for G\': C1 = {C1_Gp:.2f}, C2 = {C2_Gp:.2f}")
print(f"Arrhenius activation energy for G\': E = {E_Gp:.2f} J/mol")
print(f"WLF parameters for G\'\': C1 = {C1_Gpp:.2f}, C2 = {C2_Gpp:.2f}")
print(f"Arrhenius activation energy for G\'\': E = {E_Gpp:.2f} J/mol")
