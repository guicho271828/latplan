
-- plot various losses for meta analysis. values are normalzied per domain.

with
  mean as (
    select generator,
           avg("metrics.test.elbo") as elbo,
           avg("metrics.test.x0y0") as x0y0,
           avg("metrics.test.x1y1") as x1y1,
           avg("metrics.test.x1y2") as x1y2,
           avg("metrics.test.x0y3") as x0y3,
           avg("metrics.test.kl_z0") as kl_z0,
           avg("metrics.test.kl_z1") as kl_z1,
           avg("metrics.test.kl_a_z0") as kl_a_z0,
           avg("metrics.test.kl_a_z1") as kl_a_z1,
           avg("metrics.test.kl_z1z2") as kl_z1z2,
           avg("metrics.test.kl_z0z3") as kl_z0z3,
           avg("metrics.test.kl_z0") / N as kl_z0_per_F,
           avg("metrics.test.kl_z1z2") / N as kl_z1z2_per_F,
           avg("metrics.test.kl_z1") / N as kl_z1_per_F,
           avg("metrics.test.kl_z0z3") / N as kl_z0z3_per_F
      from ama4
     where comment = COMMENT
     group by generator
  ),
  sqmean as (
    select generator,
           avg(("metrics.test.elbo")*("metrics.test.elbo")) as elbo,
           avg(("metrics.test.x0y0")*("metrics.test.x0y0")) as x0y0,
           avg(("metrics.test.x1y1")*("metrics.test.x1y1")) as x1y1,
           avg(("metrics.test.x1y2")*("metrics.test.x1y2")) as x1y2,
           avg(("metrics.test.x0y3")*("metrics.test.x0y3")) as x0y3,
           avg(("metrics.test.kl_z0")*("metrics.test.kl_z0")) as kl_z0,
           avg(("metrics.test.kl_z1")*("metrics.test.kl_z1")) as kl_z1,
           avg(("metrics.test.kl_a_z0")*("metrics.test.kl_a_z0")) as kl_a_z0,
           avg(("metrics.test.kl_a_z1")*("metrics.test.kl_a_z1")) as kl_a_z1,
           avg(("metrics.test.kl_z1z2")*("metrics.test.kl_z1z2")) as kl_z1z2,
           avg(("metrics.test.kl_z0z3")*("metrics.test.kl_z0z3")) as kl_z0z3,
           avg(("metrics.test.kl_z0"/N)*("metrics.test.kl_z0"/N)) as kl_z0_per_F,
           avg(("metrics.test.kl_z1z2"/N)*("metrics.test.kl_z1z2"/N)) as kl_z1z2_per_F,
           avg(("metrics.test.kl_z1"/N)*("metrics.test.kl_z1"/N)) as kl_z1_per_F,
           avg(("metrics.test.kl_z0z3"/N)*("metrics.test.kl_z0z3"/N)) as kl_z0z3_per_F
      from ama4
     where comment = COMMENT
     group by generator
  ),
  std as (
    select mean.generator,
           sqrt(sqmean.elbo - mean.elbo*mean.elbo) as elbo,
           sqrt(sqmean.x0y0 - mean.x0y0*mean.x0y0) as x0y0,
           sqrt(sqmean.x1y1 - mean.x1y1*mean.x1y1) as x1y1,
           sqrt(sqmean.x1y2 - mean.x1y2*mean.x1y2) as x1y2,
           sqrt(sqmean.x0y3 - mean.x0y3*mean.x0y3) as x0y3,
           sqrt(sqmean.kl_z0 - mean.kl_z0*mean.kl_z0) as kl_z0,
           sqrt(sqmean.kl_z1 - mean.kl_z1*mean.kl_z1) as kl_z1,
           sqrt(sqmean.kl_a_z0 - mean.kl_a_z0*mean.kl_a_z0) as kl_a_z0,
           sqrt(sqmean.kl_a_z1 - mean.kl_a_z1*mean.kl_a_z1) as kl_a_z1,
           sqrt(sqmean.kl_z1z2 - mean.kl_z1z2*mean.kl_z1z2) as kl_z1z2,
           sqrt(sqmean.kl_z0z3 - mean.kl_z0z3*mean.kl_z0z3) as kl_z0z3,
           sqrt(sqmean.kl_z0_per_F - mean.kl_z0_per_F*mean.kl_z0_per_F) as kl_z0_per_F,
           sqrt(sqmean.kl_z1z2_per_F - mean.kl_z1z2_per_F*mean.kl_z1z2_per_F) as kl_z1z2_per_F,
           sqrt(sqmean.kl_z1_per_F - mean.kl_z1_per_F*mean.kl_z1_per_F) as kl_z1_per_F,
           sqrt(sqmean.kl_z0z3_per_F - mean.kl_z0z3_per_F*mean.kl_z0z3_per_F) as kl_z0z3_per_F
      from mean
             join sqmean
     where mean.generator = sqmean.generator
  ),
  valid as (
    select "parameters.generator" as generator,
           "parameters.time_start" as time_start,
           sum(valid = "true") as valid,
           count(*) as total
      from planning
     where planning.noise = 0.0
       and planning."parameters.comment" = COMMENT
       and planning."parameters.aeclass" = 'CubeSpaceAE_AMA4Conv'
     group by generator, time_start
  )
select
  -- (("metrics.test.elbo")-mean.elbo)/std.elbo as elbo,
  -- (("metrics.test.x0y0")-mean.x0y0)/std.x0y0 as x0y0,
  -- (("metrics.test.x1y1")-mean.x1y1)/std.x1y1 as x1y1,
  -- (("metrics.test.x1y2")-mean.x1y2)/std.x1y2 as x1y2,
  -- (("metrics.test.x0y3")-mean.x0y3)/std.x0y3 as x0y3,
  -- (("metrics.test.kl_z0")-mean.kl_z0)/std.kl_z0 as kl_z0,
  -- (("metrics.test.kl_z1")-mean.kl_z1)/std.kl_z1 as kl_z1,
  -- (("metrics.test.kl_a_z0")-mean.kl_a_z0)/std.kl_a_z0 as kl_a_z0,
  -- (("metrics.test.kl_a_z1")-mean.kl_a_z1)/std.kl_a_z1 as kl_a_z1,
  -- (("metrics.test.kl_z1z2")-mean.kl_z1z2)/std.kl_z1z2 as kl_z1z2,
  -- (("metrics.test.kl_z0z3")-mean.kl_z0z3)/std.kl_z0z3 as kl_z0z3,
  -- (("metrics.test.kl_z0"/N)-mean.kl_z0_per_F)/std.kl_z0_per_F as kl_z0_per_F,
  -- (("metrics.test.kl_z1z2"/N)-mean.kl_z1z2_per_F)/std.kl_z1z2_per_F as kl_z1z2_per_F,
  -- (("metrics.test.kl_z1"/N)-mean.kl_z1_per_F)/std.kl_z1_per_F as kl_z1_per_F,
  -- (("metrics.test.kl_z0z3"/N)-mean.kl_z0z3_per_F)/std.kl_z0z3_per_F as kl_z0z3_per_F,
  -- select ("metrics.test.elbo") as elbo,
  ("metrics.test.x0y0") as x0y0,
  ("metrics.test.x1y1") as x1y1,
  ("metrics.test.x1y2") as x1y2,
  ("metrics.test.x0y3") as x0y3,
  ("metrics.test.kl_z0") as kl_z0,
  ("metrics.test.kl_z1") as kl_z1,
  ("metrics.test.kl_a_z0") as kl_a_z0,
  ("metrics.test.kl_a_z1") as kl_a_z1,
  ("metrics.test.kl_z1z2") as kl_z1z2,
  ("metrics.test.kl_z0z3") as kl_z0z3,
  ("metrics.test.kl_z0")/N as kl_z0_per_F,
  ("metrics.test.kl_z1z2")/N as kl_z1z2_per_F,
  ("metrics.test.kl_z1")/N as kl_z1_per_F,
  ("metrics.test.kl_z0z3")/N as kl_z0z3_per_F,
  -- N,
  -- log(N),
  -- beta_d,
  -- log(beta_d),
  -- beta_z,
  -- log(beta_z),
  cast(valid as real)/cast(total as real) as ratio
  from ama4
         join mean
         join std
         join valid
 where ama4.generator = std.generator
   and ama4.generator = mean.generator
   and ama4."time_start" = valid.time_start
   and ama4.generator = 'latplan.puzzles.puzzle_mnist'
   and comment = COMMENT


