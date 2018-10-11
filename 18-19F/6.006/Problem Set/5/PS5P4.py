def sort_letters(ls):

    a = [0] * 26

    for l in ls:  # O(k) (the length of the longest word in the list)
        a[ord(l) - 97] += 1

    ls_ = []

    for i in range(26):  # O(26) (the range of possible letters)
        ls_ = ls_ + ([chr(i + 97)] * a[i])

    return ''.join(ls_)


def count_anagrams(strings):

    # Remove Duplicates

    strings_prime = {}

    for string in strings:  # O(n) (All the strings in the list)
        strings_prime[string] = None

    anagrams = {}

    for string in strings_prime:  # O(n) (All the strings, minus duplicates)

        sorted_string = sort_letters(string)  # O(k) (Length of longest word in the list)

        if sorted_string not in anagrams:
            anagrams[sorted_string] = []

        anagrams[sorted_string].append(string)

    anagram_pairs = 0

    for sorted_string, strings in anagrams.items():  # O(n) (If all the strings are different length)
        anagram_pairs += (len(strings) * (len(strings) - 1)) // 2  # len(strings) choose 2

    return anagram_pairs
