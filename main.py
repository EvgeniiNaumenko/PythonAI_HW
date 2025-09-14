import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from math import sqrt

np.random.seed(42)

# Завдання 1
# Потрібно проаналізувати взаємозв'язок між користувачами, сесіями та виручкою за днями. Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
# Сформуйте таблицю мінімум на 30 днів із колонками "date", "users", "sessions", "revenue".
# Розрахуйте кореляційну матрицю для цих метрик.
# Побудуйте діаграми розсіювання для пар: users-sessions, users-revenue, sessions-revenue.
# Побудуйте лінійний графік "revenue" за датами.
# Виведіть матрицю та всі графіки.

def task1():
    days = 30
    start_date = datetime.date.today() - datetime.timedelta(days=days)
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]

    users = np.random.poisson(lam=500, size=days) + np.linspace(0, 50, days).astype(int)
    sessions = np.round(users * np.random.normal(1.8, 0.1, size=days)).astype(int)
    revenue = np.round(users * np.random.normal(2.5, 0.5, size=days) + np.random.normal(50, 20, size=days), 2)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "users": users,
        "sessions": sessions,
        "revenue": revenue
    })

    print("\n=== Задание 1: первые строки таблицы ===")
    print(df.head())

    corr = df[["users", "sessions", "revenue"]].corr()
    print("\nКорреляционная матрица:\n", corr)

    plt.figure(figsize=(6,4))
    plt.scatter(df["users"], df["sessions"])
    plt.xlabel("users"); plt.ylabel("sessions")
    plt.title("Users vs Sessions")
    plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4))
    plt.scatter(df["users"], df["revenue"])
    plt.xlabel("users"); plt.ylabel("revenue")
    plt.title("Users vs Revenue")
    plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4))
    plt.scatter(df["sessions"], df["revenue"])
    plt.xlabel("sessions"); plt.ylabel("revenue")
    plt.title("Sessions vs Revenue")
    plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(df["date"], df["revenue"], marker="o")
    plt.xlabel("date"); plt.ylabel("revenue")
    plt.title("Revenue over time")
    plt.xticks(rotation=45); plt.grid(True); plt.tight_layout(); plt.show()

    return df, corr

# Завдання 2
# Потрібно проаналізувати дані A/B-експерименту та візуалізувати конверсії. Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
# Сформуйте таблицю з полями "group" (A або B) і "converted" (0/1) з не менш ніж 100 спостереженнями в кожній групі.
# Розрахуйте конверсію в групах, абсолютну різницю та відносну зміну.
# Побудуйте 95% довірчі інтервали для конверсії в кожній групі.
# Побудуйте стовпчасту діаграму конверсій груп із відображенням довірчих інтервалів.
# Виведіть усі розраховані значення та графік.

def task2(n_per_group=200):
    n = max(100, n_per_group)

    pA_true = 0.12
    pB_true = 0.15
    conv_A = np.random.binomial(1, pA_true, n)
    conv_B = np.random.binomial(1, pB_true, n)

    df = pd.DataFrame({
        "group": ["A"]*n + ["B"]*n,
        "converted": np.concatenate([conv_A, conv_B])
    })

    summary = df.groupby("group")["converted"].agg(['sum','count','mean']).rename(columns={'sum':'conversions','mean':'conversion_rate'})
    summary['conversions'] = summary['conversions'].astype(int)

    pA = summary.loc['A','conversion_rate']
    pB = summary.loc['B','conversion_rate']
    abs_diff = pB - pA
    rel_change = (pB - pA) / pA if pA != 0 else np.nan

    z = 1.96
    def prop_ci(p, n):
        se = sqrt(p*(1-p)/n)
        return (p - z*se, p + z*se)

    ci_A = prop_ci(pA, int(summary.loc['A','count']))
    ci_B = prop_ci(pB, int(summary.loc['B','count']))

    print("\n=== Задание 2: A/B эксперимент ===")
    print(summary)
    print(f"\nAbsolute difference (B - A) = {abs_diff:.4f}")
    print(f"Relative change (B vs A) = {rel_change:.2%}")
    print(f"95% CI A: ({ci_A[0]:.4f}, {ci_A[1]:.4f})")
    print(f"95% CI B: ({ci_B[0]:.4f}, {ci_B[1]:.4f})")

    groups = ['A','B']
    rates = [pA, pB]
    errs = [pA - ci_A[0], pB - ci_B[0]]

    plt.figure(figsize=(6,4))
    plt.bar(groups, rates)
    plt.errorbar(groups, rates, yerr=errs, fmt='none', capsize=6)
    plt.ylabel("Conversion rate")
    plt.title("A/B Conversion rates (95% CI)")
    plt.ylim(0, max(rates)+0.1)
    plt.grid(axis='y'); plt.tight_layout(); plt.show()

    return df, summary, (ci_A, ci_B)

# Завдання 3
# Потрібно перевірити дію центральної граничної теореми на прикладі несиметричного розподілу.
# Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
# Згенеруйте генеральну сукупність щонайменше з 50 000 спостережень із несиметричного розподілу.
# Сформуйте кілька підвибірок фіксованого розміру n і для кожної обчисліть середнє.
# Збережіть вибіркові середні та побудуйте їхню гістограму.
# Повторіть процедуру для щонайменше двох різних n і виведіть обидві гістограми.
# Виведіть середнє і стандартне відхилення вибіркових середніх для кожного n.

def task3(pop_size=50000, n_samples=2000, n_list=(5,30,100)):
    population = np.random.exponential(scale=2.0, size=pop_size)

    results = {}
    for n in n_list:
        means = np.empty(n_samples)
        for i in range(n_samples):
            idx = np.random.randint(0, pop_size, n)
            means[i] = population[idx].mean()
        mean_of_means = means.mean()
        std_of_means = means.std(ddof=1)
        print(f"\n=== Задание 3: n={n} ===")
        print(f"Среднее выборочных средних = {mean_of_means:.4f}")
        print(f"Стандартное отклонение выборочных средних = {std_of_means:.4f}")

        plt.figure(figsize=(6,4))
        plt.hist(means, bins=30)
        plt.title(f"Гистограмма выборочных средних (n={n})")
        plt.xlabel("sample mean"); plt.ylabel("frequency")
        plt.grid(True); plt.tight_layout(); plt.show()

        results[n] = {'means': means, 'mean': mean_of_means, 'std': std_of_means}

    plt.figure(figsize=(6,4))
    plt.hist(population, bins=50)
    plt.title("Распределение популяции (экспоненциальное)")
    plt.xlabel("value"); plt.ylabel("frequency")
    plt.grid(True); plt.tight_layout(); plt.show()

    return results, population

# Завдання 4
# Потрібно проаналізувати часовий ряд продажів і візуалізувати ковзаючі метрики. Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
# Сформуйте таблицю "date" і "sales" за 90 днів.
# Додайте ковзне середнє і ковзне стандартне відхилення за обраним вікном.
# Побудуйте графік вихідних продажів і графік ковзного середнього на одному полі.
# Побудуйте окремий графік ковзного стандартного відхилення.
# Виведіть таблицю з першими рядками нових стовпців і обидва графіки.


def task4(days=90, window=7):
    start = datetime.date.today() - datetime.timedelta(days=days)
    dates = pd.date_range(start=start, periods=days)

    trend = np.linspace(100, 160, days)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(days) / 7)
    noise = np.random.normal(0, 8, days)
    sales = np.round(trend + seasonality + noise, 2)

    df = pd.DataFrame({"date": dates, "sales": sales})
    df["rolling_mean"] = df["sales"].rolling(window=window, min_periods=1).mean()
    df["rolling_std"] = df["sales"].rolling(window=window, min_periods=1).std()

    print("\n=== Задание 4: первые строки с новыми столбцами ===")
    print(df.head(12))

    plt.figure(figsize=(10,4))
    plt.plot(df["date"], df["sales"], marker='o', label="sales")
    plt.plot(df["date"], df["rolling_mean"], marker='s', label=f"rolling_mean_{window}d")
    plt.xlabel("date"); plt.ylabel("sales")
    plt.title("Sales and rolling mean")
    plt.legend(); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(df["date"], df["rolling_std"], marker='o')
    plt.xlabel("date"); plt.ylabel("rolling_std")
    plt.title(f"Rolling standard deviation (window={window})")
    plt.xticks(rotation=45); plt.grid(True); plt.tight_layout(); plt.show()

    return df



df1, corr_matrix = task1()
df2, summary2, ci = task2(n_per_group=250)
results_clt, population = task3(pop_size=50000, n_samples=2000, n_list=(5,30,100))
df4 = task4(days=90, window=7)


