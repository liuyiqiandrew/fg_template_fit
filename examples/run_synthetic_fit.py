"""Run a synthetic foreground template fit example."""

from fg_template_fit.examples import example_driver


if __name__ == "__main__":
    ad, a_s = example_driver(nside=64, seed=0)
    print(f"Estimated dust amplitude ad: {ad:.6f}")
    print(f"Estimated sync amplitude as: {a_s:.6f}")
