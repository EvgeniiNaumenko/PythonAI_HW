import numpy as np

# ===== Завдання 1 =====
arr1 = np.random.randint(0, 51, 20)          # 20 випадкових чисел 0..50
print("Вихідний масив:", arr1)

threshold = int(input("Введіть порогове значення: "))
greater_count = np.sum(arr1 > threshold)
print("Кількість елементів > порогу:", greater_count)

max_val = np.max(arr1)
first_idx = np.argmax(arr1)
print(f"Максимум: {max_val}, позиція першої появи: {first_idx}")

sorted_desc = np.sort(arr1)[::-1]
print("Масив за спаданням:", sorted_desc)


# ===== Завдання 2 =====
low = int(input("Введіть нижню межу діапазону: "))
high = int(input("Введіть верхню межу діапазону: "))
matrix2 = np.random.randint(low, high + 1, (5, 5))
print("Вихідна матриця:\n", matrix2)

diag = np.diag(matrix2)
print("Головна діагональ:", diag)
print("Сума головної діагоналі:", np.sum(diag))

# обнулити елементи вище діагоналі
matrix2_mod = matrix2.copy()
matrix2_mod[np.triu_indices(5, k=1)] = 0
print("Матриця після занулення елементів вище діагоналі:\n", matrix2_mod)


# ===== Завдання 3 =====
start = int(input("Початок діапазону: "))
end = int(input("Кінець діапазону: "))
seq = np.arange(start, end + 1)
if seq.size < 30:  # на всякий випадок, щоб було 6×5
    raise ValueError("Діапазон повинен містити щонайменше 30 чисел")
matrix3 = seq[:30].reshape(6, 5)
print("Матриця 6×5:\n", matrix3)

row_sums = np.sum(matrix3, axis=1)
print("Сума по рядках:", row_sums)

col_max = np.max(matrix3, axis=0)
print("Максимуми по стовпцях:", col_max)


# ===== Завдання 4 =====
low4 = int(input("Введіть нижню межу (можна від’ємну): "))
high4 = int(input("Введіть верхню межу: "))
arr4 = np.random.randint(low4, high4 + 1, 15)
print("Вихідний масив:", arr4)

negatives = arr4[arr4 < 0]
print("Від’ємні елементи:", negatives)

arr4_mod = arr4.copy()
arr4_mod[arr4_mod < 0] = 0
print("Масив після заміни від’ємних на нулі:", arr4_mod)

zero_count = np.sum(arr4_mod == 0)
print("Кількість нульових елементів:", zero_count)


# ===== Завдання 5 =====
length = int(input("Введіть довжину масивів: "))
arr5_a = np.random.randint(0, 11, length)
arr5_b = np.random.randint(10, 21, length)
print("Перший масив:", arr5_a)
print("Другий масив:", arr5_b)

merged = np.concatenate((arr5_a, arr5_b))
print("Об’єднаний масив:", merged)

sum_arr = arr5_a + arr5_b
diff_arr = arr5_a - arr5_b
print("Поелементне додавання:", sum_arr)
print("Поелементна різниця:", diff_arr)


# ===== Завдання 6 =====
rows = int(input("Кількість рядків вихідної матриці: "))
cols = int(input("Кількість стовпців вихідної матриці: "))
matrix6 = np.arange(1, rows * cols + 1).reshape(rows, cols)
print("Вихідна матриця:\n", matrix6)

new_rows = int(input("Нова кількість рядків: "))
new_cols = int(input("Нова кількість стовпців: "))
if new_rows * new_cols != rows * cols:
    raise ValueError("Кількість елементів повинна збігатися!")
matrix6_new = matrix6.reshape(new_rows, new_cols)
print("Перетворена матриця:\n", matrix6_new)

row_min = np.min(matrix6_new, axis=1)
row_max = np.max(matrix6_new, axis=1)
print("Мінімальні по рядках:", row_min)
print("Максимальні по рядках:", row_max)

total_sum = np.sum(matrix6_new)
print("Сума всіх елементів:", total_sum)
