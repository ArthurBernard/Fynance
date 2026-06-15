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

   .. grid-item-card:: :octicon:`pin;1.2em;sd-mr-1` Position sizing
      :link: portfolio.sizing
      :link-type: doc

      Fractional Kelly, volatility targeting and transaction costs.

.. toctree::
   :maxdepth: 1
   :hidden:

   portfolio.allocation
   portfolio.sizing
