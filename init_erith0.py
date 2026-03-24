import numpy as np
import json
from typing import Any, Dict

import init_pump
from pump import pump

# Загрузка параметров клетки, концентраций ионов и проницаемостей из файла data.json
with open('data.json', 'r', encoding='utf-8') as f:
    data: Dict[str, Any] = json.load(f)

cell_params: Dict[str, float] = data['cell']

# Физические константы
F: float = cell_params['F']  # Число Фарадея, Кл/моль
R: float = cell_params['R']  # Универсальная газовая постоянная, Дж/(моль*К)
T: float = cell_params['T']  # Температура, К

# Удельная емкость мембраны, Ф/см^2
Cm: float = cell_params['Cm']

# Площадь поверхности и объем эритроцита
Ac: float = cell_params['Ac']  # Площадь мембраны, см^2 (150 мкм^2)
Vc: float = cell_params['Vc']  # Объем эритроцита, см^3 (87 мкм^3)

# Внутриклеточные концентрации ионов, моль/см^(3)
Na_i: float = cell_params['Na_i']
K_i: float = cell_params['K_i']
Mg_i: float = cell_params['Mg_i']
Ca_i: float = cell_params['Ca_i']

# Количество ионов внутри клетки, моль
n_Na: float = Na_i * Vc
n_K: float = K_i * Vc
n_Mg: float = Mg_i * Vc
n_Ca: float = Ca_i * Vc

# Внеклеточные концентрации ионов, моль/см^(3)
Na_e: float = cell_params['Na_e']
K_e: float = cell_params['K_e']
Mg_e: float = cell_params['Mg_e']
Ca_e: float = cell_params['Ca_e']

Cl_e: float = cell_params['Cl_e']
Xi_e: float = cell_params['Xi_e']
Pie: float = Na_e + K_e + Mg_e + Ca_e + Cl_e + Xi_e  # Осмотическая концентрация

# Проницаемости мембраны, см/с
PNa: float = cell_params['PNa']
PK: float = cell_params['PK']
PCl: float = cell_params['PCl']
PCa: float = cell_params['PCa']

# Параметры Ca^(2+)-зависимых K^(+)-каналов
PKmax: float = cell_params['PKmax']
Nch: float = cell_params['Nch']
Kch: float = cell_params['Kch']

# Проницаемость Ca^(2+)-зависимых K^(+)-каналов
Pch: float = PKmax * (n_Ca / (n_Ca + Vc * Kch)) ** Nch

# Параметры Ca^(2+)-АТФазы
K_CaATP: float = cell_params['K_CaATP']

# Расчёт равновесного состояния клетки
alp: float = 3 * (PK + Pch) / (2 * PNa)

# Начальный мембранный потенциал
Em0: float = -R * T * np.log((Na_i + alp * K_i) / (Na_e + alp * K_e)) / F

Cl_i: float = Cl_e * np.exp((Em0 * F) / (R * T))
n_Cl: float = Cl_i * Vc

# Осмотические концентрации
Xi_i: float = Pie - Na_i - K_i - Mg_i - Ca_i - Cl_i
n_Xi: float = Xi_i * Vc

Pii: float = (n_Na + n_K + n_Mg + n_Ca + n_Cl + n_Xi) / Vc

# Эффективный заряд органических анионов
zx: float = (Ac * Cm * Em0 / F - n_Na - n_K - 2 * n_Mg - 2 * n_Ca + n_Cl) / n_Xi

# Расчет потоков и параметров насоса
u_val: float = F * Em0 / (R * T)

# Ограничение u для предотвращения переполнения
u_val = float(np.clip(u_val, -10.0, 10.0))

if abs(u_val) >= 0.00001:
    epsm: float = u_val / (np.exp(0.5 * u_val) - np.exp(-0.5 * u_val))
    epsm2: float = 2 * u_val / (np.exp(u_val) - np.exp(-u_val))
else:
    epsm = 1.0
    epsm2 = 1.0

# Вызов функции pump
Ap: float = pump(u_val, Na_i, Na_e, K_i, K_e,
                 init_pump.k12, init_pump.k23, init_pump.k34o, init_pump.k45,
                 init_pump.k56, init_pump.k61, init_pump.k21, init_pump.k32,
                 init_pump.k43o, init_pump.k54, init_pump.k65, init_pump.k16,
                 init_pump.ATP, init_pump.ADP, init_pump.P_i)

# Электродиффузионные потоки
JNa: float = PNa * epsm * (Na_e * np.exp(-0.5 * u_val) - Na_i * np.exp(0.5 * u_val))
JK: float = (PK + Pch) * epsm * (K_e * np.exp(-0.5 * u_val) - K_i * np.exp(0.5 * u_val))
JCa: float = PCa * epsm2 * (Ca_e * np.exp(-u_val) - Ca_i * np.exp(u_val))

# Коэффициенты для Na^(+)/K^(+)-АТФазы
N: float = (JNa / (3 * Ap))
NN: float = (-JK / (2 * Ap))  # Должны совпадать

# Ca^(2+)-АТФаза
Q_CaATP: float = 0.5 * JCa / (Ca_i / (Ca_i + K_CaATP)) ** 2

print(f"alp = {alp}")
print(f"Em0 = {Em0}")
print(f"N = {N}")
print(f"NN = {NN}")
print(f"Q_CaATP = {Q_CaATP}")
