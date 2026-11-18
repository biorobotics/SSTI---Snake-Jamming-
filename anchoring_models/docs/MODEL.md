# Equations, signs and assumptions

## Initial anchoring

Given geometry q0, measured joint torque and friction coefficient mu, use the
existing zero-external-wrench Min-Max optimization.

\[
\min_{\mathbf F_n,\mathbf F_t,\mathbf R,z\ge0}z,
\qquad F_{ni}^2+F_{ti}^2\le z,\quad i=1,2,3.
\]

\[
F_{n1}-F_{n2}+R_n=0,\quad F_{t1}-F_{t2}+R_t=0,
\]
\[
F_{n3}-R_n=0,\quad F_{t3}-R_t=0,
\]
\[
hF_{n1}-\ell_LF_{t1}=\tau_0,\qquad
hF_{n3}+\ell_RF_{t3}=\tau_0,
\]
\[
F_{ni}\ge0,\qquad -\mu F_{ni}\le F_{ti}\le\mu F_{ni}.
\]

The native reaction variables Rn and Rt correspond to rx and ry in the original
solver. They act with opposite signs on the two rigid bodies. Eliminating them
gives the two force equations in the compact figure. The epigraph formulation
above is equivalent to minimizing the maximum squared contact resultant.

The default geometry preserves the existing conversion.

\[
\phi=(\pi-q_0)/2,\quad h=L\cos\phi+2r,\quad
\ell_L=\ell_R=L\sin\phi,\quad \tau_0=|\tau_{\mathrm{feedback},0}|.
\]

The caller supplies q0 after the appropriate instrument conversion. The
defaults are L=0.1021 m and r=0.0167 m. Explicit asymmetric arms are supported.
Initialization returns the complete optimized six-force state and independent
residual checks. An infeasible solver result raises `RuntimeError("INFEASIBLE")`.

## Complete initial state and shear overwrite

After optimization, store the complete initial normal and shear state.

\[
\mathbf F_{n0}=\mathbf F_n^*,\qquad\mathbf F_{t0}=\mathbf F_t^*.
\]

At each reaching snapshot, normals receive the prescribed increment and shear
is assigned by the current load-sharing rule, replacing the initial shear.
Thus the initial shear is a preserved optimization result. Zero horizontal
reaction produces zero shear when the reaching rule is applied. This overwrite
is a provisional response assumption. Its two-body moment consistency is
assessed separately from the original optimum.

## Current contact-1 transfer rule

Let A=Fn1,0 and C=Fn3,0. At each snapshot, the signed applied reaction
increments are Fh and Fv at ATI136.

\[
N_1=A-F_v,\quad N_3=C+F_v,\quad N_2=F_{n2,0}+F_v.
\]

The native initial force balance gives Fn2,0=A+C to solver tolerance. The
implementation retains each optimized normal value independently.

\[
w_1=\frac{N_1}{N_1+N_3},\quad w_3=\frac{N_3}{N_1+N_3},\qquad
\mathbf T=-\frac{F_h}{2}[w_1,1,w_3]^T.
\]

Vertical loading contributes zero shear. These equations reproduce the current
45-degree replay after gauge-baseline subtraction, with Fh=Fv=p/sqrt(2).
N and T denote the predicted normal and common-frame shear vectors.

## Preserved reaching site-linear rule

For an applied reaction at Rpre1, decrease N1 by Fv. At Rnext2, decrease N3 by
Fv. The other initial normal components remain at their frozen values.

\[
\mathbf T=-F_h[1/4,1/2,1/4]^T.
\]

This is the existing fixed-share reaching rule. In both response modes, the
common shear signs obey sum(Ti)+Fh=0 and the normal balance obeys
N1-N2+N3+Fv=0 to the initial solver tolerance. The modes retain separate
application-site contracts.

## Signs and contact tracking

\[
\mathbf F_t^{\mathrm{common}}=
[F_{t1}^{\mathrm{native}},-F_{t2}^{\mathrm{native}},F_{t3}^{\mathrm{native}}]^T.
\]

The original optimizer uses native local signs. Response CSVs and plots use
common shear signs. Normal force is compressive-positive. The optional
measurement columns must already use these plotting conventions.

\[
m_i=\mu N_i-|T_i|,\qquad u_i=|T_i|/(\mu N_i).
\]

With zero configured force tolerance, warn if any Ni<=0, or any positive-normal
contact has mi<0. A positive `force_tolerance_N` applies the declared numerical
threshold to both comparisons. Utilization is undefined for a lost contact or
a zero friction denominator and is written as null in snapshot output.

The first alarm latches. Repeating the same reaction at the next time sample
produces the same force demand. Inputs are evaluated from the initial state
without history accumulation, filtering, interpolation, clipping or cone
projection. Time must increase strictly.

The applied couple M is reported as `unmapped_applied_moment_Nm`. The response
does not enforce full two-body moment equilibrium. The result field
`full_two_body_moment_equilibrium_verified` records this modeling limitation.

## Instantaneous reaching dynamics

The fixed-J1 planar rod model uses the existing Lagrange mass matrix and
Newton–Euler inverse dynamics.

\[
M(q)\ddot q+c(q,\dot q)=\tau,\qquad
\ddot q=M(q)^{-1}(\tau-c(q,\dot q)).
\]

\[
M(q)=\sum_k[m_kJ_{v,k}^TJ_{v,k}+I_kJ_{\omega,k}^TJ_{\omega,k}],
\qquad I_k=m_kL_k^2/12.
\]

\[
\mathbf F_{\mathrm{robot\ on\ base}}=-\sum_km_k\mathbf a_{c,k},\qquad
M_{\mathrm{robot\ on\ base}}=-\sum_k[I_k\alpha_k+
\mathbf r_{c,k}\times(m_k\mathbf a_{c,k})].
\]

The base is fixed and in-plane gravity is zero. Six-link defaults are 0.5 kg and
0.11 m per link. q, dq and feedback torque follow the same named-axis mask,
in physical base-to-tip order. The zero pose points link 1 along +X. A configured
yaw rotates force components into the assumed anchor frame. This coordinate
rotation retains the fixed-base origin for the planar moment.

## Symbol definitions

| Symbol | Meaning | Unit |
|---|---|---|
| i=1,2,3 | Contact index, ATI136/157/158 | unitless |
| n, t subscripts | Normal and tangential contact components | — |
| 0, * | Initial reference and optimizer output | — |
| Fn, Ft | Three-component contact-force vectors | N |
| Rn, Rt | Internal joint-reaction components in native solver axes | N |
| q0 | Converted anchor geometry angle | rad |
| h, ellL, ellR | Geometry-derived moment arms | m |
| L, r | Anchor link length and radius | m |
| phi | Auxiliary geometry angle | rad |
| tau0 | Magnitude of initial measured joint feedback torque | N m |
| z | Upper bound on each squared contact resultant | N² |
| mu | Coulomb friction coefficient | unitless |
| Wext,0 | Initial external planar wrench, set to zero | N, N, N m |
| t | Time during reaching or replay | s |
| freach=(Fh,Fv) | Signed applied reaction force in the selected anchor frame | N |
| p | Gauge reading after baseline subtraction | N |
| Delta Fn | Assumed normal-force response increment | N |
| w1, w3 | Current normal-force-weighted shares within the 1/3 wall | unitless |
| Ni, Ti | Updated normal and common-frame shear predictions | N |
| mi, ui | Friction margin and utilization | N, unitless |
| M | Applied external planar couple | N m |
| q, dq, ddq | Reaching joint angle, velocity, acceleration | rad, rad/s, rad/s² |
| M(q), c | Mass matrix and Coriolis/centrifugal torque vector | kg m², N m |
| mk, Lk, Ik | Reaching-link mass, length, uniform-rod inertia | kg, m, kg m² |
| Jv, Jomega | COM and angular Jacobians | m, unitless |
| ac, rc, alpha | COM acceleration, COM position from J1, link angular acceleration | m/s², m, rad/s² |
| bold type, superscript T | Vector or matrix, transpose | — |
| abs bars, exists, or | Absolute value, at least one contact, logical disjunction | — |
| min, max, s.t. | Minimize, largest value, subject to | — |
