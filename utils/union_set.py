class UnionSet:
    """并查集实现"""
    def __init__(self, length: int):
        self.__len = length
        self.__set_count = length
        self.__parent = list(range(length))
        self.__rank = [1] * length

    def find(self, x: int) -> int:
        if self.__parent[x] != x:
            self.__parent[x] = self.find(self.__parent[x])
        return self.__parent[x]

    def union(self, i: int, j: int) -> bool:
        """返回值表示 i j 是否属于同一集合"""
        x, y = self.find(i), self.find(j)
        if x == y:
            return True
        self.__set_count -= 1
        if self.__rank[x] <= self.__rank[y]:
            self.__parent[x] = y
        else:
            self.__parent[y] = x
        if self.__rank[x] == self.__rank[y]:
            self.__rank[y] += 1
        return False

    def get_set_count(self) -> int:
        return self.__set_count

    def get_sets(self) -> dict:
        ret = {}
        for i, gid in enumerate(self.__parent):
            if gid not in ret:
                ret[gid] = []
            ret[gid].append(i)
        return ret

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, item: int) -> int:
        return self.__parent[item]

    def __iter__(self) -> iter:
        return iter(self.__parent)
