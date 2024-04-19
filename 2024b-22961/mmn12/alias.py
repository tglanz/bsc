# https://link.springer.com/chapter/10.1007/978-1-4842-7185-8_21

from random import randint, random
from typing import List, Tuple
from matplotlib import pyplot as plt

class AliasTable:

    table: List[Tuple[float, int, int]]

    def __init__(self, distribution: List[float]):
        self.table = AliasTable.construct(distribution)

    def sample(self) -> float:
        uniform_bin = randint(0, len(self.table) - 1) 
        epsilon = random()
        tau, i, j = self.table[uniform_bin]
        return i if epsilon < tau else j

    @staticmethod
    def construct(distribution: List[float]) -> List[Tuple[float, int, int]]:

        N = len(distribution)
        w = sum(distribution) / N

        remaining = list(enumerate(distribution))

        table = []
        
        while remaining:
            # sort by weight
            remaining.sort(key=lambda e: e[1])
            print(remaining)
            
            # i is the sample and wi is the weight of the sample
            i, wi = remaining[0]
            
            tau = wi / w        

            # j is the highest weighted sample
            j = remaining[-1][0]

            wj = remaining[-1][1]

            remaining[-1] = (j, 2*wj - w)

            # append an entry to the table
            table.append((tau, i, j))

            # remove sample i
            remaining.pop(0)

        return table

def main():
    distribution = [.2, .1, .4, .3]
    alias = AliasTable(distribution)

    samples = 10000
    counters = {}
    for _ in range(samples):
        val = alias.sample()
        counters[val] = counters.get(val, 0) + 1

    for k, v in counters.items():
        counters[k] = (v, f"{round(v / samples * 100, 2)}%")

    print(counters)

if __name__ == '__main__':
    main()
