def bsp_value(L, m):
    n = len(L)

    # Sort the list in ascending order
    L.sort()

    # Initialize a 2D array to store optimal solutions to subproblems
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Populate the dynamic programming table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Optimal solution considering removing j elements up to index i
            dp[i][j] = max(dp[i - 1][j], L[i - 1] - L[i - 2] + dp[i - 2][j - 1])

    # The maximum value is found in the last entry of the table
    return dp[n][m]


# Example usage:
L = [2, 4, 6, 7, 10, 14]
m = 2
result = bsp_value(L, m)
print(f"Maximum value: {result}")

def bsp_solution(L, m):
    n = len(L)

    # Sort the list in ascending order
    L.sort()

    # Initialize a 2D array to store optimal solutions to subproblems
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Populate the dynamic programming table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Optimal solution considering removing j elements up to index i
            dp[i][j] = max(dp[i - 1][j], L[i - 1] - L[i - 2] + dp[i - 2][j - 1])

    # Reconstruct the solution
    solution = []
    i, j = n, m
    while i > 0 and j > 0:
        # If the current element is included in the optimal solution
        if dp[i][j] != dp[i - 1][j]:
            solution.append(L[i - 1])
            i -= 2
            j -= 1
        else:
            i -= 1

    return sorted(solution)

# Example usage:
L = [2, 4, 6, 7, 10, 14]
m = 2
result = bsp_solution(L, m)
print(f"Optimized solution: {result}")
