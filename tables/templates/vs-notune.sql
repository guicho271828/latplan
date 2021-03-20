with
  x as (
    select *
      from AECLASS
     where comment = 'kltune2'),
  y as (
    select *
      from AECLASS
     where comment = 'notune2')
select generators.id,
       x.metric,
       x.is_best,
       y.metric,
       y.is_best
  from generators
         join (
           select generator, "metrics.test.elbo" as metric, beta_z,beta_d,N,
                  ("metrics.test.elbo" in (select min("metrics.test.elbo")
                                     from x
                                    group by generator, N)) as is_best
             from x
         ) as x
             on generators.key = x.generator
         cross join (
           select generator, "metrics.test.elbo" as metric, beta_z,beta_d,N,
                  0 as is_best
             from y
         ) as y
             on generators.key = y.generator
             and x.N = y.N
