def mergesort(L):
    if len(L) <= 1:
        return L
    num = int(len(L)/2)
    left = mergesort(L[:num])
    right = mergesort(L[num:])
    return merge(left, right)


def merge(left, right):
    r, l = 0, 0
    result = []
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += right[r:]
    result += left[l:]
    return result


if __name__ == '__main__':
    L = [4,3,6,9,7,1,2,4]
print(mergesort(L))