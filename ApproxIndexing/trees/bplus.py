import pickle

order = 4


class Splitter:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Node:
    def __init__(self, tree, parent=None):
        self.tree = tree
        self.parent = parent
        self.left_sibling = None
        self.right_sibling = None
        self.entries = {}
        self.leaf = True

    def insert(self, row):
        if self.leaf:
            if row[1] in self.entries.keys():
                self.entries[row[1]].append(row)
            else:
                self.entries[row[1]] = [row]

                if len(self.entries) == order:
                    self.split()
        else:
            sortedkeys = list(self.entries.keys())
            sortedkeys = sorted(sortedkeys)

            chosenkey = None

            for key in sortedkeys:

                if row[1] < key:
                    chosenkey = self.entries[key].left
                    break
                else:
                    chosenkey = self.entries[key].right

            chosenkey.insert(row)

    def split(self):

        keys = list(self.entries.keys())
        keys = sorted(keys)

        midkey_position = int(len(keys)/2)
        midkey = keys[midkey_position]

        if self.parent is None:

            if self.leaf:
                left = Node(self.tree, parent=self)
                left.entries = {x: self.entries[x] for x in keys if x < midkey}


                right = Node(self.tree, parent=self)
                right.entries = {x: self.entries[x] for x in keys if x >= midkey}

                left.right_sibling = right
                right.left_sibling = left

                splitter = Splitter(left, right)
                self.entries = {midkey: splitter}
                self.leaf = False

            else:
                new_parent = Node(self.tree)
                new_parent.leaf = False

                left = Node(self.tree, parent=new_parent)
                left.entries = {x: self.entries[x] for x in keys if x < midkey}
                if not self.leaf:
                    for entry in left.entries.values():
                        entry.left.parent = left
                        entry.right.parent = left

                left.leaf = self.leaf

                right = Node(self.tree, parent=new_parent)
                if self.leaf:
                    right.entries = {x: self.entries[x] for x in keys if x >= midkey}
                else:
                    right.entries = {x: self.entries[x] for x in keys if x > midkey}
                    for entry in right.entries.values():
                        entry.left.parent = right
                        entry.right.parent = right

                right.leaf = self.leaf

                left.right_sibling = right
                right.left_sibling = left

                splitter = Splitter(left, right)
                new_parent.entries = {midkey: splitter}
                self.tree.switch_root(new_parent)

        else:
            parentkeys = list(self.parent.entries.keys())
            parentkeys = sorted(parentkeys)

            # Find parent siblings.
            parent_leftsibling = None
            parent_rightsibling = None
            for key in parentkeys:
                if key < midkey:
                    parent_leftsibling = self.parent.entries[key]
                if key > midkey:
                    parent_rightsibling = self.parent.entries[key]
                    break

            left = Node(self.tree, self.parent)
            left.entries = {x: self.entries[x] for x in keys if x < midkey}
            if not self.leaf:
                for entry in left.entries.values():
                    entry.left.parent = left
                    entry.right.parent = left
            left.leaf = self.leaf

            right = Node(self.tree, self.parent)
            if self.leaf:
                right.entries = {x: self.entries[x] for x in keys if x >= midkey}
            else:
                right.entries = {x: self.entries[x] for x in keys if x > midkey}
                for entry in right.entries.values():
                    entry.left.parent = right
                    entry.right.parent = right

            right.leaf = self.leaf

            left.right_sibling = right
            right.left_sibling = left

            splitter = Splitter(left, right)
            self.parent.entries[midkey] = splitter

            # Relink the existing splitters to the new nodes.
            parentkeys = list(self.parent.entries.keys())
            parentkeys = sorted(parentkeys)
            lefttoentry = parentkeys[parentkeys.index(midkey)-1] \
                if (parentkeys.index(midkey)-1) >= 0 else None
            righttoentry = parentkeys[parentkeys.index(midkey)+1] \
                if (parentkeys.index(midkey)+1) < len(self.parent.entries) else None

            if lefttoentry is not None:
                self.parent.entries[lefttoentry].right = splitter.left

            if righttoentry is not None:
                self.parent.entries[righttoentry].left = splitter.right

            # Acquire new siblings.
            try:
                parent_leftsibling.left.right_sibling = left
                left.left_sibling = parent_leftsibling.left
            except Exception as e:
                print("Debug")

            try:
                parent_rightsibling.right.left_sibling = right
                right.right_sibling = parent_rightsibling.right
            except:
                pass

            if len(self.parent.entries) == order:
                self.parent.split()

    def query(self, q=None, autocomplete=False):
        resultids = []
        number = int(q.right)

        if not self.leaf:
            sortedkeys = list(self.entries.keys())
            sortedkeys = sorted(sortedkeys)

            chosenNode = None

            for key in sortedkeys:
                if number < key:
                    chosenNode = self.entries[key].left
                    break
                else:
                    chosenNode = self.entries[key].right

            resultids = chosenNode.query(q)

            return resultids

        else:

            if not autocomplete:
                valid_keys = [x for x in self.entries.keys() if eval('x' + q.operator + q.right)]

                result = []

                for key in valid_keys:
                    result.extend([x[0] for x in self.entries[key]])

            else:
                result = [item[0] for sublist in self.entries.values() for item in sublist]


            if (q.operator == ">=") | (q.operator == ">"):
                if self.right_sibling is not None:
                    siblingresult = self.right_sibling.query(q, True)
                    result.extend(siblingresult)
                    return result
                else:
                    return result

            if (q.operator == "<=") | (q.operator == "<"):
                if self.left_sibling is not None:
                    siblingresult = self.left_sibling.query(q, True)
                    result.extend(siblingresult)
                    return result
                else:
                    return result

    def get_data(self):
        if not self.leaf:
            data = []
            for entry in self.entries.values():
                data.extend(entry.left.get_data())
                data.extend(entry.right.get_data())

            return data
        else:
            data = []
            for entry in self.entries.values():
                data.extend(entry)

            return data

class Tree:
    def __init__(self, data):
        self.root = Node(self)

        self.populate(data)

    def query(self, q):
        result = self.root.query(q)
        return result

    def populate(self, data):
        for row in data:
            self.root.insert(row)

    def switch_root(self, node):
        self.root = node

    def get_data(self):
        result = self.root.get_data()
        result = set(result)

        return result