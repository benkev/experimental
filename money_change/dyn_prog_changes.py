#
# dyn_prog_changes.py
#
# Determine the number of ways to make change for 1 ruble
# using coins of 50, 20, 10, 5, 2, and 1 kopeck.
#

coins = [50, 20, 10, 5, 2, 1]
target = 100  # 1 рубль в копейках # 1 ruble in kopecks

#
# Инициализация массива динамического программирования
# The dp array stores the number of ways to make change for each amount
# from 0 to 100.
#
dp = [0] * (target + 1)
# dp[0] = 1  # 1 способ разменять 0 копеек
dp[0] = 1  # There is one way to make zero kopecks (use no coins)

# Основной алгоритм
# Compute ways to make change
for coin in coins:
    for amount in range(coin, target + 1):
        dp[amount] += dp[amount - coin]

# Результат
print(f"Количество способов разменять 1 рубль: {dp[target]}")
print(f"Number of ways to make change for 1 ruble: {dp[target]}")
