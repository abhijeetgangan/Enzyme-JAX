"""Nested derivatives of a Lennard-Jones system from jax-md.

Both tests differentiate a scalar loss with respect to the potential
parameters sigma and epsilon. The loss already contains second derivatives
of the energy, so the forward and reverse passes tested here are third-order
AD overall:

  LJHessian: dense Hessian, one Hessian-vector product per basis vector.
    Under jit the loop over the 3N basis vectors unrolls into the jaxpr,
    which makes compile time the bottleneck.
  LJElastic: athermal elastic moduli, second derivatives of the energy plus
    an implicit solve for the non-affine response.
"""

from functools import partial

from absl.testing import absltest

from test_utils import EnzymeJaxTest


def lj_system(n_rep):
    """Simple cubic lattice in fractional coordinates plus its neighbor list."""
    import numpy as np
    import jax.numpy as jnp
    from jax_md import space, energy
    from jax_md.util import f64

    lattice_constant = 1.37820
    R = jnp.array(np.array(list(np.ndindex((n_rep,) * 3))) / n_rep, f64)
    box = jnp.array(np.eye(3) * n_rep * lattice_constant, f64)

    displacement, _ = space.periodic_general(box)
    neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
        displacement, box, fractional_coordinates=True
    )
    nbrs = neighbor_fn.allocate(R)
    return R, box, energy_fn, nbrs


class LJHessian(EnzymeJaxTest):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        from jax_md.util import f64

        R, box, energy_fn, nbrs = lj_system(2)

        def hessian(fn, R):
            _, hvp = jax.linearize(jax.grad(fn), R)
            basis = jnp.eye(R.size).reshape(-1, *R.shape)
            return jnp.stack([hvp(e) for e in basis]).reshape(R.shape + R.shape)

        def loss(params):
            fn = partial(energy_fn, neighbor=nbrs, sigma=params[0], epsilon=params[1])
            return jnp.sum(hessian(fn, R) ** 2)

        self.fn = loss
        self.name = "lj_hessian" + str(R.shape[0])
        self.repeat = 2
        self.ins = [jnp.array([1.0, 1.0], f64)]
        self.dins = [jnp.array([1.0, 1.0], f64)]
        self.douts = jnp.ones((), f64)
        self.atol = 1e-6
        self.rtol = 1e-6


class LJElastic(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp
        from jax_md import elasticity
        from jax_md.util import f64

        R, box, energy_fn, nbrs = lj_system(2)
        emt_fn = elasticity.athermal_moduli(energy_fn)

        def loss(params):
            C = emt_fn(R, box, neighbor=nbrs, sigma=params[0], epsilon=params[1])
            return jnp.sum(C**2)

        self.fn = loss
        self.name = "lj_elastic" + str(R.shape[0])
        self.repeat = 2
        self.ins = [jnp.array([1.0, 1.0], f64)]
        self.dins = [jnp.array([1.0, 1.0], f64)]
        self.douts = jnp.ones((), f64)
        self.atol = 1e-6
        self.rtol = 1e-6
        # Enzyme MLIR AD cannot differentiate the conjugate-gradient solve in
        # athermal_moduli yet: the stablehlo.while has no static iteration
        # count ("WhileOp does not have known iteration count for cache
        # removal"). Primal and PostRev still run through all pipelines.
        self.mlirad_fwd = False
        self.mlirad_rev = False


if __name__ == "__main__":
    import platform

    # Deps not available on macos
    if platform.system() != "Darwin" and platform.machine() == "x86_64":
        from test_utils import fix_paths

        fix_paths()
        import jax

        jax.config.update("jax_enable_x64", True)
        absltest.main()
