import numpy as np
from causallens.data.eda import balance_table


def test_binary_encoding_invariant_and_known_proportion_difference():
    X=np.tile(np.array([0.,0.,0.,1.,0.,1.,1.,1.])[:,None],(1,25))
    T=np.array([0]*4+[1]*4)
    table=balance_table(X,T)
    np.testing.assert_allclose(table.smd, .5/np.sqrt(.1875))
    X[:,13]+=1
    np.testing.assert_allclose(balance_table(X,T).smd,table.smd)


def test_numeric_signed_smd_and_no_mutation():
    X=np.tile(np.array([0.,1.,2.,3.,2.,3.,4.,5.])[:,None],(1,25))
    before=X.copy();T=np.array([0]*4+[1]*4)
    np.testing.assert_allclose(balance_table(X,T).smd,2/np.sqrt(5/3))
    np.testing.assert_array_equal(X,before)
    np.testing.assert_allclose(balance_table(X,1-T).smd,-2/np.sqrt(5/3))
