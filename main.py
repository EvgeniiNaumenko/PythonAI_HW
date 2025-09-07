import numpy as np

# ===== № 1 =====
arr1 = np.random.randint(0, 51, 20)
print("Массив:", arr1)

threshold = int(input("Введите порогове значение: "))
greater_count = np.sum(arr1 > threshold)
print("Больше > порога:", greater_count)

max_val = np.max(arr1)
first_idx = np.argmax(arr1)
print(f"Максимум: {max_val}, 1я позиция: {first_idx}")

sorted_desc = np.sort(arr1)[::-1]
print("Массив на убівание:", sorted_desc)


# ===== № 2 =====
low = int(input("Введите нижнюю границу: "))
high = int(input("Введите верхнюю границу: "))
matrix2 = np.random.randint(low, high + 1, (5, 5))
print("Матрица:\n", matrix2)

diag = np.diag(matrix2)
print("Главная диагональ:", diag)
print("Сумма главной диагонали:", np.sum(diag))

matrix2_mod = matrix2.copy()
matrix2_mod[np.triu_indices(5, k=1)] = 0
print("матрица после обнуления диагонали:\n", matrix2_mod)


# ===== № 3 =====
start = int(input("Начало диапазона: "))
end = int(input("Конец диапазона: "))
seq = np.arange(start, end + 1)
if seq.size < 30:
    raise ValueError("Диапазон минимум 30")
matrix3 = seq[:30].reshape(6, 5)
print("Матрица 6×5:\n", matrix3)

row_sums = np.sum(matrix3, axis=1)
print("Сумма по строках:", row_sums)

col_max = np.max(matrix3, axis=0)
print("Максимумы по столбцах:", col_max)


# ===== № 4 =====
low4 = int(input("Введите нижнюю границу: "))
high4 = int(input("Введите верхнюю границу: "))
arr4 = np.random.randint(low4, high4 + 1, 15)
print("Массив:", arr4)

negatives = arr4[arr4 < 0]
print("Отрицательные элементы:", negatives)

arr4_mod = arr4.copy()
arr4_mod[arr4_mod < 0] = 0
print("Отрицательные на нули:", arr4_mod)

zero_count = np.sum(arr4_mod == 0)
print("Количестку нулевых:", zero_count)


# ===== № 5 =====
length = int(input("Введите длину массивов: "))
arr5_a = np.random.randint(0, 11, length)
arr5_b = np.random.randint(10, 21, length)
print("Первый масив:", arr5_a)
print("ДругийВторой масив:", arr5_b)

merged = np.concatenate((arr5_a, arr5_b))
print("Обьедененный массив:", merged)

sum_arr = arr5_a + arr5_b
diff_arr = arr5_a - arr5_b
print("Сумма:", sum_arr)
print("Разница:", diff_arr)


# ===== № 6 =====
rows = int(input("Количество строк: "))
cols = int(input("Количество столбцов: "))
matrix6 = np.arange(1, rows * cols + 1).reshape(rows, cols)
print("Матрица:\n", matrix6)

new_rows = int(input("Новое количество строк: "))
new_cols = int(input("Новое количество столбцов: "))
if new_rows * new_cols != rows * cols:
    raise ValueError("Не совпадает количество элементов")
matrix6_new = matrix6.reshape(new_rows, new_cols)
print("Матрица:\n", matrix6_new)

row_min = np.min(matrix6_new, axis=1)
row_max = np.max(matrix6_new, axis=1)
print("Минимум по строках:", row_min)
print("МаксимМаксимум по строках:", row_max)

total_sum = np.sum(matrix6_new)
print("Сумма:", total_sum)
