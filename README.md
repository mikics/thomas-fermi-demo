# MPI issue

The issue is reproduces in `demo_tf_mwe.py`, where the original problem has been
extremely simplified.

The problem is the following one: whenever you run `demo_tf_mwe.py` with
1 processor, the solution is fine. Whenever you use a number of processors `np >=2`,
the solution is full of `inf+nanj` (see the last line 
`print(Esh_rz_m.x.array[:]`).

## Other useful information:
- The problem seems to be caused by the following term in the weak form:

```
ufl.inner(curl_Es_m, curl_v_m) * rho * dDom

```

- The problem seems to be caused by `curl_r` and `curl_p` in `curl_axis`, and
therefore `curl_z` has been set to 0:

```
curl_r = (-a[2].dx(1) - 1j * m / rho * a[1])
curl_p = (a[0].dx(1) - a[1].dx(0))
```

- In the full problem, with `m=0` the solution is calculated correctly whatever
the number of processors. This behaviour cannot be shown in the `mwe` since
we get an error (probably for `m=0` too many terms are get rid of in the `mwe`,
and therefore PETSc cannot calculate the solution).
