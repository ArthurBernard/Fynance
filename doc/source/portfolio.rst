--------------------------------------------------
 Portfolio (:mod:`fynance.portfolio`)
--------------------------------------------------

Portfolio allocation algorithms and rolling walk-forward wrappers.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 0
   :padding: 0

   .. grid-item-card:: :octicon:`project;1.2em;sd-mr-1` Portfolio allocation
      :link: portfolio.allocation
      :link-type: doc

      Equal Risk Contribution (ERC), Hierarchical Risk Parity (HRP),
      Inverse Variance Portfolio (IVP), Maximum Diversified Portfolio
      (MDP), Minimum Variance Portfolio (MVP, MVP_uc) and a rolling
      walk-forward wrapper.

   .. grid-item-card:: :octicon:`list-ordered;1.2em;sd-mr-1` Risk decomposition
      :link: portfolio.attribution
      :link-type: doc

      Marginal and absolute risk contributions of assets to portfolio
      volatility, with causal rolling decomposition.

   .. grid-item-card:: :octicon:`shield-lock;1.2em;sd-mr-1` Constraint projection
      :link: portfolio.constraints
      :link-type: doc

      Least-distance projection of weights onto a box, gross-leverage
      cap, net-exposure range and named group bounds.

   .. grid-item-card:: :octicon:`pin;1.2em;sd-mr-1` Position sizing
      :link: portfolio.sizing
      :link-type: doc

      Fractional Kelly, volatility targeting and transaction costs.

   .. grid-item-card:: :octicon:`graph;1.2em;sd-mr-1` Conditioned covariance
      :link: portfolio.covariance
      :link-type: doc

      Sample covariance, Ledoit-Wolf shrinkage, exponentially weighted,
      factor-model and Marchenko-Pastur denoised estimators.

.. toctree::
   :maxdepth: 1
   :hidden:

   portfolio.allocation
   portfolio.attribution
   portfolio.constraints
   portfolio.covariance
   portfolio.sizing
