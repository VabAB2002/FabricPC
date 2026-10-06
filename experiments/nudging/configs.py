"""pcx's VGG-5 nudging settings, copied from their yaml files.

Source: github.com/liukidar/pcx, branch build-docs,
examples/s4_1_discriminative_mode/VGG5/{cifar10,cifar100,tinyimagenet}/
VGG5_PCN_{SE,PN,NN}.yaml. Targets are Table 1 of arXiv 2407.01163 v2.

pcx has no code or yaml for centered nudging (CN). Its paper says CN
alternates positive and negative nudging epochs at random, so the CN
configs reuse each dataset's NN settings with a random sign per epoch.
"""

import dataclasses
from typing import Dict, Tuple


@dataclasses.dataclass(frozen=True)
class NudgeConfig:
    name: str
    dataset: str  # "cifar10", "cifar100" or "tinyimagenet"
    num_classes: int
    input_shape: Tuple[int, int, int]
    T: int  # settling steps per batch
    beta0: float
    factor: float  # +1 positive / plain, -1 negative
    beta_ir: float  # beta grows by this much per epoch
    lr_x: float  # latent step size (SGD with momentum)
    momentum: float
    lr_w: float  # AdamW base rate
    wd: float
    target: float  # pcx's published accuracy, in %
    centered: bool = False  # draw factor = +1/-1 at random each epoch
    epochs: int = 50
    batch_size: int = 128


_C10 = dict(dataset="cifar10", num_classes=10, input_shape=(32, 32, 3))
_C100 = dict(dataset="cifar100", num_classes=100, input_shape=(32, 32, 3))
_TINY = dict(dataset="tinyimagenet", num_classes=200, input_shape=(56, 56, 3))

_C10_NN = dict(
    T=7,
    beta0=0.17877062530418192,
    factor=-1.0,
    beta_ir=0.02,
    lr_x=0.028016192698375384,
    momentum=0.5,
    lr_w=0.000261833244376427,
    wd=0.0008992487435200682,
)
_C100_NN = dict(
    T=12,
    beta0=0.15,
    factor=-1.0,
    beta_ir=0.02,
    lr_x=0.03325378866925405,
    momentum=0.25,
    lr_w=7.67519702690274e-05,
    wd=0.004202804226490381,
)

CONFIGS: Dict[str, NudgeConfig] = {
    c.name: c
    for c in [
        NudgeConfig(
            name="c10-se",
            **_C10,
            T=7,
            beta0=1.0,
            factor=1.0,
            beta_ir=0.0,
            lr_x=0.02348681827141059,
            momentum=0.3,
            lr_w=0.0002657691155688603,
            wd=0.0066559627972820395,
            target=87.98,
        ),
        NudgeConfig(
            name="c10-pn",
            **_C10,
            T=7,
            beta0=0.32297201469734277,
            factor=1.0,
            beta_ir=0.02,
            lr_x=0.03623356587254243,
            momentum=0.1,
            lr_w=0.00029968925891952494,
            wd=3.728852110891977e-05,
            target=88.42,
        ),
        NudgeConfig(name="c10-nn", **_C10, **_C10_NN, target=88.83),
        NudgeConfig(name="c10-cn", **_C10, **_C10_NN, target=89.47, centered=True),
        NudgeConfig(name="c100-nn", **_C100, **_C100_NN, target=65.46),
        NudgeConfig(name="c100-cn", **_C100, **_C100_NN, target=67.19, centered=True),
        NudgeConfig(
            name="tiny-nn",
            **_TINY,
            T=8,
            beta0=0.5,
            factor=-1.0,
            beta_ir=0.0,
            lr_x=0.0573599016772709,
            momentum=0.1,
            lr_w=0.0001376789472968569,
            wd=1.6046037514265988e-05,
            target=46.40,
        ),
    ]
}
