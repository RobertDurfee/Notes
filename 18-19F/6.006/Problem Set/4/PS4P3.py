class BST:
    def __init__(self, item=None, parent=None):
        "Initialize a BST node"
        self.item = item
        self.parent = parent
        self.left = None
        self.right = None

    def _is_empty(self):
        return self.item is None

    def _min_node(self):
        "Return node with minimum item key in subtree (assumes nonempty)"
        if self.left:
            return self.left._min_node()
        return self

    def _find_node(self, k):
        "Return node from subtree having item with key k (assumes nonempty)"
        if k == self.item.key:
            return self
        if k < self.item.key and self.left:
            return self.left._find_node(k)
        if k > self.item.key and self.right:
            return self.right._find_node(k)
        return None

    def _close_node(self, k):
        '''
        Return node from (nonempty) subtree having either:
            - item with smallest key >= query key k
            - item with largest  key <= query key k
            - no item if node is empty
        '''
        if k < self.item.key and self.left:
            return self.left._close_node(k)
        if k > self.item.key and self.right:
            return self.right._close_node(k)
        return self

    def _successor_node(self):
        "Return next node in an in-order traversal of tree (assumes nonempty)"
        if self.right:
            return self.right._min_node()
        node = self
        while node.parent and (node.parent.right is node):
            node = node.parent
        return node.parent  # Will be none if self contains max item in tree

    def _replace(self, node):
        "Replace self's attributes with node's attributes"
        self.item = node.item
        self.left = node.left
        self.right = node.right
        if self.left:   self.left.parent = self
        if self.right:  self.right.parent = self

    def _delete_node(self):
        "Remove self's item from subtree (assumes nonempty)"
        node = self
        if self.left and self.right:  # has two children
            node = self.right._min_node()
            self.item = node.item
        if node.right:
            node._replace(node.right)  # has one child
        elif node.left:
            node._replace(node.left)
        else:  # has no children
            if node.parent is None:
                node.item = None
                return
            if node.parent.right is node:
                node.parent.right = None
            else:
                node.parent.left = None
            node = node.parent
        node._maintain()

    def _maintain(self):
        '''
        Perform maintenance after a dynamic operation
        Called by lowest node with subtree modified by insert or delete
        '''
        pass

    def find_min(self):
        "Return an item with minimum key, else None"
        if self._is_empty(): return None
        node = self._min_node()
        return node.item if node else None

    def find(self, k):
        "Return an item with key k, else None"
        if self._is_empty(): return None
        node = self._find_node(k)
        return node.item if node else None

    def find_next(self, k):
        "Return an item with smallest key greater than k, else None"
        if self._is_empty(): return None
        node = self._close_node(k)  # guarenteed to have item
        if node.item.key < k:
            node = node._successor_node()
        return node.item if node else None

    def insert(self, x):
        "Insert item into self's subtree"
        if self._is_empty():
            self.item = x
            self._maintain()
        elif x.key < self.item.key:
            if self.left is None:
                self.left = self.__class__(None, self)
            self.left.insert(x)
        else:
            if self.right is None:
                self.right = self.__class__(None, self)
            self.right.insert(x)

    def delete(self, k):
        "Delete key k from self's subtree"
        if self._is_empty():
            raise IndexError('delete from empty tree')
        node = self._find_node(k)
        if node is None:
            return None
        item = node.item
        node._delete_node()
        return item

    def delete_min(self):
        "Delete item with minimum key from self's subtree"
        if self._is_empty():
            raise IndexError('delete from empty tree')
        node = self._min_node()
        item = node.item
        node._delete_node()
        return item

    def iter_recursive(self):
        "Return iterator of subtree's in-order traversal (recursive)"
        if self.left:
            yield from self.left.iter_recursive()
        yield self.item
        if self.right:
            yield from self.right.iter_recursive()

    def iter_iterative(self):
        "Return iterator of subtree's in-order traversal (iterative)"
        node = self._min_node()
        while node:
            yield node.item
            node = node._successor_node()

    def _item_str(self):
        return str(self.item.key)

    def __str__(self):
        "Return ASCII drawing of the tree"
        s = self._item_str()
        if self.left is None and self.right is None:
            return s
        sl, sr = [''], ['']
        if self.left:
            s = '_' + s
            sl = str(self.left).split('\n')
        if self.right:
            s = s + '_'
            sr = str(self.right).split('\n')
        wl, cl = len(sl[0]), len(sl[0].lstrip(' _'))
        wr, cr = len(sr[0]), len(sr[0].rstrip(' _'))
        a = [(' ' * (wl - cl)) + ('_' * cl) + s +
             ('_' * cr) + (' ' * (wr - cr))]
        for i in range(max(len(sl), len(sr))):
            ls = sl[i] if i < len(sl) else ' ' * wl
            rs = sr[i] if i < len(sr) else ' ' * wr
            a.append(ls + ' ' * len(s) + rs)
        return '\n'.join(a)


class AVL(BST):
    def __init__(self, item=None, parent=None):
        "Augment BST with height and skew"
        super().__init__(item, parent)
        self.height = 0
        self.skew = 0

    def _update(self):
        "Update height and skew"
        left_height = self.left.height if self.left else -1
        right_height = self.right.height if self.right else -1
        self.height = max(left_height, right_height) + 1
        self.skew = right_height - left_height

    def _right_rotate(self):
        '''
        Rotate left to right, assuming left is not None
         __s__      __n__
        _n_  c  =>  a  _s_
        a b            b c
        Self and node swap contents so that top node stays at top
        '''
        node, c = self.left, self.right
        a, b = node.left, node.right
        self.item, node.item = node.item, self.item
        if a:   a.parent = self
        if c:   c.parent = node
        self.left, self.right = a, node
        node.left, node.right = b, c
        node._update()
        self._update()

    def _left_rotate(self):
        '''
        Rotate right to left, assuming right is not None
        __s__        __n__
        a  _n_  =>  _s_  c
           b c      a b
        Self and node swap contents so that top node stays at top
        '''
        a, node = self.left, self.right
        b, c = node.left, node.right
        self.item, node.item = node.item, self.item
        if a:   a.parent = node
        if c:   c.parent = self
        self.left, self.right = node, c
        node.left, node.right = a, b
        node._update()
        self._update()

    def _maintain(self):
        "Update height and skew and rebalance up the tree"
        self._update()
        if self.skew == 2:  # must have right child
            if self.right.skew == -1:
                self.right._right_rotate()
            self._left_rotate()
        elif self.skew == -2:  # must have left child
            if self.left.skew == 1:
                self.left._left_rotate()
            self._right_rotate()
        if self.parent:
            self.parent._maintain()

    def _item_str(self):
        return str(self.item.key) + (
            '=' if self.skew == 0 else
            '>' if self.skew < 0 else
            '<')


class TransactionLog(AVL):
    def __init__(self, item=None, parent=None):

        super().__init__(item, parent)

        self.minimum = self.item.key if item else None
        self.maximum = self.item.key if item else None
        self.sum = self.item.key if item else None

    def _update(self):

        super()._update()

        self.minimum = self.left.minimum if self.left else self.item.key
        self.maximum = self.right.maximum if self.right else self.item.key
        self.sum = self.item.d + (self.left.sum if self.left else 0) + (self.right.sum if self.right else 0)

    def add_transaction(self, x):

        super().insert(x)

    def interval_revenue(self, t1, t2):

        if t2 < self.minimum or self.maximum < t1:
            return 0

        if t1 <= self.minimum and self.maximum <= t2:
            return self.sum

        return (self.item.d if t1 <= self.item.key <= t2 else 0) + \
               (self.left.interval_revenue(t1, t2) if self.left else 0) + \
               (self.right.interval_revenue(t1, t2) if self.right else 0)

    def _item_str(self):
        return str(self.item.key) + ',' + str(self.item.d) + ',' + str(self.minimum) + ',' + str(self.maximum) + ',' + \
               str(self.sum)
