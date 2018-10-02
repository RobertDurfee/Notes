def allocate(n):
    return [None] * n


class DoubleSidedDynamicArray:

    def __init__(self):

        self.next_right = 0
        self.next_left = -1
        self.size = 1
        self.array = allocate(self.size)

    def len(self):

        return self.next_right - self.next_left - 1

    def resize_right(self, new_size):

        old_array = self.array
        self.size = new_size
        self.array = allocate(self.size)

        for i in range(self.next_left + 1, self.next_right):
            self.array[i] = old_array[i]

    def resize_left(self, new_size):

        old_array = self.array
        self.next_right = self.next_right - self.size + new_size
        self.next_left = self.next_left - self.size + new_size
        self.size = new_size
        self.array = allocate(self.size)

        for i in range(self.next_left - self.size + 1, self.next_right - self.size):
            self.array[i] = old_array[i]

    def insert_right(self, x):

        if self.next_right == self.size:
            self.resize_right(self.size + self.len())

        self.array[self.next_right] = x
        self.next_right += 1

    def insert_left(self, x):

        if self.next_left == -1:
            self.resize_left(self.size + self.len())

        self.array[self.next_left] = x
        self.next_left -= 1

    def delete_right(self):
        if self.size - self.next_right >= 3 * self.len():
            self.resize_right(self.size // 2)

        self.next_right -= 1
        self.array[self.next_right] = None

    def delete_left(self):
        if self.next_left + 1 >= 3 * self.len():
            self.resize_left(self.size // 2)

        self.next_left += 1
        self.array[self.next_left] = None
