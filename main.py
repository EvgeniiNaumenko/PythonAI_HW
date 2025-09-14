import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from math import sqrt

np.random.seed(42)  # для відтворюваності результатів

# Завдання 1
# Потрібно проаналізувати взаємозв'язок між користувачами, сесіями та виручкою за днями. Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
# Сформуйте таблицю мінімум на 30 днів із колонками "date", "users", "sessions", "revenue".
# Розрахуйте кореляційну матрицю для цих метрик.
# Побудуйте діаграми розсіювання для пар: users-sessions, users-revenue, sessions-revenue.
# Побудуйте лінійний графік "revenue" за датами.
# Виведіть матрицю та всі графіки.

days = 30
start_date = datetime.date.today() - datetime.timedelta(days=days)
dates = [start_date + datetime.timedelta(days=i) for i in range(days)]

users = np.random.poisson(lam=500, size=days) + np.linspace(0, 50, days).astype(int)
sessions = (users * np.random.normal(1.8, 0.1, size=days)).astype(int)
revenue = np.round(users * np.random.normal(2.5, 0.5, size=days) + np.random.normal(50, 20, size=days), 2)

df1 = pd.DataFrame({
    "date": pd.to_datetime(dates),
    "users": users,
    "sessions": sessions,
    "revenue": revenue
})

print("=== Завдання 1: Кореляційна матриця ===")
print(df1[["users", "sessions", "revenue"]].corr(), "\n")

plt.scatter(df1["users"], df1["sessions"])
plt.xlabel("users"); plt.ylabel("sessions")
plt.title("Users vs Sessions"); plt.grid(True)
plt.show()

plt.scatter(df1["users"], df1["revenue"])
plt.xlabel("users"); plt.ylabel("revenue")
plt.title("Users vs Revenue"); plt.grid(True)
plt.show()

plt.scatter(df1["sessions"], df1["revenue"])
plt.xlabel("sessions"); plt.ylabel("revenue")
plt.title("Sessions vs Revenue"); plt.grid(True)
plt.show()

plt.plot(df1["date"], df1["revenue"], marker="o")
plt.xlabel("date"); plt.ylabel("revenue")
plt.title("Revenue over 30 days")
plt.xticks(rotation=45); plt.grid(True)
plt.show()


# Завдання 2
# Потрібно проаналізувати дані A/B-експерименту та візуалізувати конверсії. Усе необхідно запрограмувати в Python
# з використанням pandas, NumPy і Matplotlib.
# Сформуйте таблицю з полями "group" (A або B) і "converted" (0/1) з не менш ніж 100 спостереженнями в кожній групі.
# Розрахуйте конверсію в групах, абсолютну різницю та відносну зміну.
# Побудуйте 95% довірчі інтервали для конверсії в кожній групі.
# Побудуйте стовпчасту діаграму конверсій груп із відображенням довірчих інтервалів.
# Виведіть усі розраховані значення та графік.

n = 250
conv_A = np.random.binomial(1, 0.12, n)
conv_B = np.random.binomial(1, 0.15, n)

df2 = pd.DataFrame({
    "group": ["A"]*n + ["B"]*n,
    "converted": np.concatenate([conv_A, conv_B])
})

summary = df2.groupby("group")["converted"].agg(['sum','count','mean'])
summary.rename(columns={'sum':'conversions','mean':'conversion_rate'}, inplace=True)

pA, pB = summary.loc['A','conversion_rate'], summary.loc['B','conversion_rate']
abs_diff = pB - pA
rel_change = (pB - pA) / pA * 100

def proportion_ci(p, n, z=1.96):
    se = sqrt(p*(1-p)/n)
    return (p - z*se, p + z*se)

ci_A = proportion_ci(pA, n)
ci_B = proportion_ci(pB, n)

print("=== Завдання 2: A/B-експеримент ===")
print(summary, "\n")
print(f"Absolute diff (B-A): {abs_diff:.4f}")
print(f"Relative change: {rel_change:.2f}%")
print(f"95% CI A: {ci_A}")
print(f"95% CI B: {ci_B}\n")

rates = [pA, pB]
errs = [pA-ci_A[0], pB-ci_B[0]]
plt.bar(['A','B'], rates)
plt.errorbar(['A','B'], rates, yerr=errs, fmt='none', capsize=5, color="black")
plt.ylabel("Conversion rate")
plt.title("A/B Conversion rates with 95% CI")
plt.grid(axis="y")
plt.show()


# Завдання 3
# Потрібно перевірити дію центральної граничної теореми на прикладі несиметричного розподілу.
# Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
# Згенеруйте генеральну сукупність щонайменше з 50 000 спостережень із несиметричного розподілу.
# Сформуйте кілька підвибірок фіксованого розміру n і для кожної обчисліть середнє.
# Збережіть вибіркові середні та побудуйте їхню гістограму.
# Повторіть процедуру для щонайменше двох різних n і виведіть обидві гістограми.
# Виведіть середнє і стандартне відхилення вибіркових середніх для кожного n.


population = np.random.exponential(scale=2.0, size=50000)

for n in [5, 30, 100]:
    means = [population[np.random.randint(0, len(population), n)].mean() for _ in range(2000)]
    means = np.array(means)
    print(f"=== Завдання 3: n={n} ===")
    print(f"Mean of sample means = {means.mean():.4f}, Std = {means.std(ddof=1):.4f}\n")
    plt.hist(means, bins=30)
    plt.title(f"Sample means distribution (n={n})")
    plt.xlabel("mean value"); plt.ylabel("frequency")
    plt.grid(True)
    plt.show()

plt.hist(population, bins=50)
plt.title("Population distribution (Exponential)")
plt.xlabel("value"); plt.ylabel("frequency")
plt.grid(True)
plt.show()


# Завдання 4
# Потрібно проаналізувати часовий ряд продажів і візуалізувати ковзаючі метрики.
# Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
# Сформуйте таблицю "date" і "sales" за 90 днів.
# Додайте ковзне середнє і ковзне стандартне відхилення за обраним вікном.
# Побудуйте графік вихідних продажів і графік ковзного середнього на одному полі.
# Побудуйте окремий графік ковзного стандартного відхилення.
# Виведіть таблицю з першими рядками нових стовпців і обидва графіки.


days4 = 90
dates4 = pd.date_range(datetime.date.today()-datetime.timedelta(days=days4), periods=days4)

trend = np.linspace(100, 160, days4)
seasonality = 10 * np.sin(2*np.pi*np.arange(days4)/7)
noise = np.random.normal(0, 8, days4)
sales = np.round(trend + seasonality + noise, 2)

df4 = pd.DataFrame({"date": dates4, "sales": sales})
df4["rolling_mean"] = df4["sales"].rolling(window=7, min_periods=1).mean()
df4["rolling_std"] = df4["sales"].rolling(window=7, min_periods=1).std()

print("=== Завдання 4: перші рядки таблиці ===")
print(df4.head(12), "\n")

plt.plot(df4["date"], df4["sales"], label="sales")
plt.plot(df4["date"], df4["rolling_mean"], label="rolling mean (7d)")
plt.xlabel("date"); plt.ylabel("sales")
plt.title("Sales and rolling mean")
plt.legend(); plt.xticks(rotation=45); plt.grid(True)
plt.tight_layout(); plt.show()

plt.plot(df4["date"], df4["rolling_std"])
plt.xlabel("date"); plt.ylabel("rolling std")
plt.title("Rolling standard deviation (7d)")
plt.xticks(rotation=45); plt.grid(True)
plt.tight_layout(); plt.show()
