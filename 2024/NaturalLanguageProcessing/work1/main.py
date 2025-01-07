"""
实现计算编辑距离的算法,使其在输出最小编辑距离的同时，输出具有最小编辑距离的对齐关系。
注意：本算法中替换操作与插入和删除都是cost为1的
"""

def get_edit_distance(str1, str2):
    # 初始化编辑距离矩阵
    m = len(str1) + 1
    n = len(str2) + 1
    distance = [[0] * n for _ in range(m)]

    # 初始化第一行和第一列
    for i in range(m):
        distance[i][0] = i
    for j in range(n):
        distance[0][j] = j

    # 填充编辑距离矩阵
    for i in range(1, m):
        for j in range(1, n):
            if str1[i-1] == str2[j-1]:
                distance[i][j] = min(distance[i-1][j-1], distance[i][j-1] + 1, distance[i-1][j] + 1)
            else:
                distance[i][j] = min(distance[i-1][j-1] + 1, distance[i][j-1] + 1, distance[i-1][j] + 1)

    # 回溯找到最小编辑距离的编辑路径
    i, j = m - 1, n - 1
    aligned_str1 = []
    aligned_str2 = []

    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            aligned_str1.append(str1[i - 1])
            aligned_str2.append(str2[j - 1])
            i -= 1
            j -= 1
        else:
            if distance[i][j] == distance[i - 1][j - 1] + 1:
                # 替换操作
                aligned_str1.append(str1[i - 1])
                aligned_str2.append(str2[j - 1])
                i -= 1
                j -= 1
            elif distance[i][j] == distance[i][j - 1] + 1:
                # 插入操作
                aligned_str1.append('-')
                aligned_str2.append(str2[j - 1])
                j -= 1
            else:
                # 删除操作
                aligned_str1.append(str1[i - 1])
                aligned_str2.append('-')
                i -= 1

    # 处理剩余的字符
    while i > 0:
        aligned_str1.append(str1[i - 1])
        aligned_str2.append('-')
        i -= 1
    while j > 0:
        aligned_str1.append('-')
        aligned_str2.append(str2[j - 1])
        j -= 1

    # 反转对齐后的字符串
    aligned_str1.reverse()
    aligned_str2.reverse()

    # 将对齐结果转换为字符串
    alignment_str1 = ''.join(aligned_str1)
    alignment_str2 = ''.join(aligned_str2)

    # 返回最小编辑距离和对齐关系
    return distance[m - 1][n - 1], (alignment_str1, alignment_str2)

if __name__ == '__main__':
    while True:
        # 输入两个字符串
        str1 = str(input("请输入第一个字符串："))
        str2 = str(input("请输入第二个字符串："))

        edit_distance, alignment = get_edit_distance(str1, str2)
        print("两个字符串的最小编辑距离为：", edit_distance)

        print("最小编辑距离的编辑路径为：",alignment)
