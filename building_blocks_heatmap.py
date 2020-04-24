import os

from plot_heatmap_of_differences import plot_heatmap


def differences(global_min, global_max):
    def encoding_length_all():
        return 5

    def encoding_length_exactly(n1, n2):
        return 12 + n1 + n2

    return {(n1, n2): (encoding_length_exactly(n1, n2) - encoding_length_all())
            for n1 in range(global_min, global_max + 1)
            for n2 in range(n1, global_max + 1)}


if __name__ == '__main__':
    minimum_n, maximum_n = 1, 20
    plot_heatmap(
        False,
        minimum_n, maximum_n,
        "$\\left|T^{EX}\\left(n_1, n_2 \\right)\\right| - \\left|T^{ALL}\\right|$" + "\nLower is Better",
        os.path.join("figures", "bb_len_diff.png"),
        maximum_n, differences(minimum_n, maximum_n))
