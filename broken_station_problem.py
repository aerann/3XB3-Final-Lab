def bsp_value(L, m):
    n = len(L)

    # Initialize a variable to store the maximum value
    max_value = float('-inf')

    # Iterate through the sorted list and calculate differences
    for i in range(n - m + 1):
        max_value = max(max_value, L[i + m - 1] - L[i])

    return max_value

def bsp_solution(L, m):
    n = len(L)
    the_bsp_value = bsp_value(L,m)
    final_list = []

    # Iterate through the sorted list and calculate differences
    for i in range(n):
        current_num = L[i]
        for j in range(i+1, n):
            current_diff = L[j] - L[i]
            if current_diff == the_bsp_value:
                final_list.append(L[i])
                final_list.append(L[j])

    final_unique_list = list(set(final_list))
    final_unique_list.sort()
    return final_unique_list

# Example usage:
L = [1, 2, 7, 8, 12, 17]
m = 2
result = bsp_value(L, m)
print(f"Maximum value: {result}")
result = bsp_solution(L, m)
print(f"Optimized solution: {result}")
