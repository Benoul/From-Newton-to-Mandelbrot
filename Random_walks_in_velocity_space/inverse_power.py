import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SEED = 100
A = 1.0
p = 1.0
n_density = 0.5
v_rel = 5.0  
b_min = 0.5
b_max = 20.0
dt = 0.1
T = 200.0

def inverse_power(A, p, n_density, v_rel, b_min, b_max, dt, T, SEED=100):
    np.random.seed(SEED)

    cross_section = np.pi * (b_max**2 - b_min**2)
    encounter_rate = n_density * v_rel * cross_section
    num_steps = int(T / dt)

    times = np.linspace(0, T, num_steps+1)
    vx = np.zeros(num_steps+1)
    vy = np.zeros(num_steps+1)
    vz = np.zeros(num_steps+1)
    speed = np.zeros(num_steps+1)

    vx[0] = v_rel
    vy[0] = 0.0
    vz[0] = 0.0
    speed[0] = np.sqrt(v_rel**2)

    kick_magnitudes = []

    for i in range(num_steps):

        mean_k = encounter_rate * dt
        k = np.random.poisson(mean_k)

        v_vec = np.array([vx[i], vy[i], vz[i]])
        v_norm = np.linalg.norm(v_vec)

        vx_new, vy_new, vz_new = v_vec

        for _ in range(k):
            u = np.random.rand()
            b = np.sqrt(b_min**2 + u * (b_max**2 - b_min**2))

            dv_mag = A/ (v_rel * (b**(p-1)))
            kick_magnitudes.append(dv_mag)

            if v_norm == 0:
                perp = np.random.normal(size=3)
                perp /= np.linalg.norm(perp)
            else:
                rand_vec = np.random.normal(size=3)
                perp = rand_vec - (np.dot(rand_vec, v_vec) / (v_norm**2)) * v_vec
                perp /= np.linalg.norm(perp)

            dv = dv_mag * perp
            vx_new += dv[0]
            vy_new += dv[1]
            vz_new += dv[2]

        vx[i+1] = vx_new
        vy[i+1] = vy_new
        vz[i+1] = vz_new
        speed[i+1] = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2)


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(vx, vy, linewidth=0.7)
    plt.scatter(vx[0], vy[0], label='start', color = 'k',zorder=5)
    plt.scatter(vx[-1], vy[-1], label='end', color = 'r',zorder=5)
    plt.xlabel("vx")
    plt.ylabel("vy")
    plt.title(f"Velocity Space Random Walk (vx vs vy), inverse-power-law (p={p})")
    plt.legend()
    plt.grid()


    def func(x, m, c, k):
        return m * (x **k) + c

    (m, c, k), _ = curve_fit(func, times, speed**2)
    time_fit = np.linspace(min(times), max(times), 100)
    speed_fit = func(time_fit, m, c, k)
    
    plt.subplot(1, 2, 2)
    plt.plot(times, speed**2, marker = 'o', label='Data')
    plt.plot(time_fit, speed_fit, label = f'Slope = {m}, Power = {k:.2f}', color='r')
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("speed squared")
    plt.title(f"Speed Squared vs Time, inverse-power-law (p={p})")
    plt.grid()

    plt.tight_layout()
    plt.show()

    return vx, vy, vz, speed, times, kick_magnitudes
