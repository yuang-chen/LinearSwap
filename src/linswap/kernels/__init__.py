"""Built-in kernels.  Importing this package registers them."""

from . import deltanet, gdn, gdn2, gla, kda, mamba2, mamba3, rwkv7, swa  # noqa: F401  (mamba3 registers only with its kernels installed)
