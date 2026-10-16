import re
from argparse import ArgumentParser
from collections.abc import Callable, MutableMapping
from copy import deepcopy
from itertools import product

from rich.console import Console
from rich.table import Table

console = Console()


class DiscreteFunction:
    def __init__(self, func: str):
        self._func = self.parse_function(func)
        self._len = len(self._func)

    def __iter__(self):
        return iter(self._func)

    def __str__(self):
        output = ""
        for i in range(self._len):
            miniterm = self._func[i]
            for t in miniterm:
                if t[0] == "+":
                    output += f"x{t[1]}"
                else:
                    output += f"!x{t[1]}"
            if i < self._len - 1:
                output += " v "
        return output

    @staticmethod
    def parse_function(func: str) -> list[list[tuple[str, int]]]:
        pattern = r"(!?)x(\d+)"
        res = []
        for string in func.split(" v "):
            miniterm = []
            for match in re.finditer(pattern, string):
                sign = "-" if match.group(1) == "!" else "+"
                num = int(match.group(2))
                miniterm.append((sign, num))
            res.append(miniterm)

        return res


class Miniterm(MutableMapping):
    def __init__(self, miniterm: list[tuple[str, int]]):
        self._miniterm = self.create_miniterm(miniterm)

    def __getitem__(self, key):
        return self._miniterm[key]

    def __setitem__(self, key, value):
        self._miniterm[key] = value

    def __delitem__(self, key):
        del self._miniterm[key]

    def __iter__(self):
        return iter(self._miniterm)

    def __len__(self):
        return len(self._miniterm)

    def __repr__(self):
        output = ""
        for k in self._miniterm:
            if self._miniterm[k](1):
                output += f"x{k}"
            else:
                output += f"!x{k}"

        return output

    def __str__(self):
        output = ""
        for k in self._miniterm:
            if self._miniterm[k](1):
                output += f"x{k}"
            else:
                output += f"!x{k}"

        return output

    @staticmethod
    def create_miniterm(
        func: list[tuple[str, int]],
    ) -> dict[int, Callable[[int], bool | int]]:
        miniterm = {}
        for sign, n in func:
            miniterm[n] = (lambda x: x) if sign == "+" else (lambda x: not x)
        return miniterm


class MinitermList:
    def __init__(self, miniterms: list[Miniterm]):
        self._miniterms = miniterms

    def __iter__(self):
        return iter(self._miniterms)

    def __repr__(self):
        return " v ".join([str(miniterm) for miniterm in self._miniterms])

    def __str__(self):
        return " v ".join([str(miniterm) for miniterm in self._miniterms])

    @staticmethod
    def normalize_op(miniterm_list: MinitermList, n: int, xvalue: int) -> MinitermList:
        new_op = []
        copy_func = deepcopy(miniterm_list)
        for miniterm in copy_func:
            if n in miniterm:
                if miniterm[n](xvalue):
                    del miniterm[n]
                    new_op.append(miniterm)
            else:
                new_op.append(miniterm)

        return MinitermList(new_op)

    def differentiate(
        self, miniterm_list: MinitermList, n: int
    ) -> tuple[MinitermList, MinitermList]:
        left_op = self.normalize_op(miniterm_list, n, 0)
        right_op = self.normalize_op(miniterm_list, n, 1)

        return left_op, right_op


class XORTable:
    def __init__(self, left_op: MinitermList, right_op: MinitermList):
        """
        _fv (function values) - значения функции при данных аргументах
        """
        self._left_op = left_op
        self._right_op = right_op
        self._fv = XORTable.create_cascade_table(left_op, right_op)
        self._weight = sum(self._fv.values())
        self._table = XORTable.create_beautiful_table(
            self._fv, self._weight, left_op, right_op
        )

    @property
    def get_table(self):
        return self._fv

    @property
    def get_weight(self):
        return self._weight

    def show_table(self):
        console.print(self._table)

    @staticmethod
    def create_beautiful_table(
        data: dict[tuple, bool],
        weight: int,
        left_op: MinitermList,
        right_op: MinitermList,
    ):
        args = XORTable.get_args(left_op, right_op)
        table = Table(
            title="Cascade Table", caption=f"Weight (P): {weight}", show_lines=True
        )
        for arg in sorted(args):
            table.add_column(f"x{arg}", style="blue")
        table.add_column("Function Result", style="green")

        for args, res in data.items():
            args = [str(i) for i in args]
            table.add_row(*args, str(res))

        return table

    @staticmethod
    def create_cascade_table(
        left_op: MinitermList, right_op: MinitermList
    ) -> dict[tuple, int]:
        args = XORTable.get_args(left_op, right_op)
        sorted_args = sorted(args)
        k = len(args)

        table = {}
        for t in product((0, 1), repeat=k):
            dvalues = {k: v for k, v in zip(sorted_args, t)}
            func_res = XORTable.calculate_list_of_miniterms(
                left_op, dvalues
            ) ^ XORTable.calculate_list_of_miniterms(right_op, dvalues)
            table[t] = func_res

        return table

    @staticmethod
    def get_args(left_op: MinitermList, right_op: MinitermList) -> set[int]:
        res = set()
        for miniterm in left_op:
            res |= set(miniterm.keys())
        for miniterm in right_op:
            res |= set(miniterm.keys())

        return res

    @staticmethod
    def calculate_list_of_miniterms(
        op: list[Miniterm], dvalues: dict[int, int]
    ) -> bool:
        for miniterm in op:
            miniterm_args = set(miniterm.keys())
            if all(miniterm[k](v) for k, v in dvalues.items() if k in miniterm_args):
                return True

        return False


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="DiscreteCalculator",
        description="Calculates derivatives for a discrete function",
    )
    parser.add_argument("function")

    args = parser.parse_args()
    func = DiscreteFunction(args.function)

    for i in range(1, 8):
        miniterm_list = MinitermList(
            [Miniterm(miniterm_list) for miniterm_list in func]
        )
        left_op, right_op = miniterm_list.differentiate(miniterm_list, i)
        table = XORTable(left_op, right_op)
        table.show_table()
