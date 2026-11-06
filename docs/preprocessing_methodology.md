# Preprocessing methodology

Planned: preserve immutable raw inputs; establish reproducible splits before learning transformations; fit imputation, encoding, and scaling only on training data; apply unchanged transforms to held-out data. Exclude counterfactual/oracle outputs from estimator inputs. Record seeds, split indices, transformation parameters, schema assertions, and output checksums. Exact choices remain pending inspection.
