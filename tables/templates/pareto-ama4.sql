
-- plot reconstruction losses vs kl divergences

select (("metrics.test.x0y0"+("metrics.test.x1y1"+"metrics.test.x1y2")/2) +
        ("metrics.test.x1y1"+("metrics.test.x0y0"+"metrics.test.x0y3")/2) )/2 as reconstruction,
       (("metrics.test.kl_z0"+"metrics.test.kl_a_z0"+"metrics.test.kl_z1z2"/2) +
        ("metrics.test.kl_z1"+"metrics.test.kl_a_z1"+"metrics.test.kl_z0z3"/2))/2 as kl,
       N, beta_z, beta_d,
       (cast(N as text) || ' ' || cast(beta_z as text) || ' ' || cast(beta_d as text)) as label
  from ama4
 where generator = GENERATOR
   and comment = 'kltune2'
