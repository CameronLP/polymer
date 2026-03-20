from matplotlib import pyplot as plt
import pytest
from polymer.main_v5 import run_polymer_dataset
from eoread.pace import get_sample, Level1B_PACE_OCI
from core.tests.conftest import savefig
from core.tests.graphics import xrimshow


@pytest.mark.parametrize("sample", [1, 2])
def test_pace_reader(request, sample):
    sample_level1 = get_sample(sample)

    l1 = Level1B_PACE_OCI(sample_level1["path"])

    l1.Rtoa.sel(bands=849).isel(**sample_level1["roi"]).plot()

    if "px" in sample_level1:
        px = sample_level1["px"]
        plt.plot([px["x"]], [px["y"]], "r+")

    savefig(request)


@pytest.mark.parametrize('uncertainties', [True, False])
@pytest.mark.parametrize("sample", [1, 2])
def test_pace_polymer(request, sample, uncertainties: bool):
    product_level1 = get_sample(sample)

    l1 = Level1B_PACE_OCI(product_level1["path"])

    l2 = run_polymer_dataset(l1, uncertainties=uncertainties).sel(product_level1["roi"])

    plt.figure()
    _, ax, _ = xrimshow(l2.rho_w.sel(bands=500, method="nearest"), vmin=0)
    ax.plot([product_level1['px']['x']], [product_level1['px']['y']], 'ro')
    savefig(request)

    if uncertainties:
        plt.figure()
        xrimshow(l2.rho_w_unc.sel(bands=500, method="nearest"), vmin=0)
        ax.plot([product_level1['px']['x']], [product_level1['px']['y']], 'ro')
        savefig(request)


@pytest.mark.parametrize("sample", [1, 2])
def test_pace_polymer_singlepixel(request, sample):
    product_level1 = get_sample(sample)
    l1 = Level1B_PACE_OCI(product_level1["path"])

    if "px" not in product_level1:
        return

    y = product_level1["roi"]["y"].start + product_level1["px"]["y"]
    x = product_level1["roi"]["x"].start + product_level1["px"]["x"]
    l2 = run_polymer_dataset(
        l1.sel(
            y=slice(y, y + 1),
            x=slice(x, x + 1),
        ),
        uncertainties=True,
    ).isel(x=0, y=0).compute()

    l2 = l2.rename(Rprime='rho_rc')

    for var, c, label in [
        # ("Rtoa", "b", "Rtoa"),
        # ("rho_gc", "g", "rho_gc"),
        ("Ratm", "g", 'Ratm'),
        ("rho_rc", "b", 'rho_rc'),
        ("rho_w", "k", 'rho_w (± uncertainty)'),
    ]:
        l2[var].plot(label=label, color=c)  # type: ignore
    
    (l2['rho_w'] + l2['rho_w_unc']/2).plot(ls='--', color='k') # type: ignore
    (l2['rho_w'] - l2['rho_w_unc']/2).plot(ls='--', color='k') # type: ignore

    plt.plot(
        l2.bands_corr, [0 for _ in l2.bands_corr], "r.", label="bands used by Polymer AC"
    )

    plt.legend()
    plt.grid(True)
    savefig(request)
