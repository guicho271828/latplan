with
  x as (
    select *
      from ama4
     where comment = 'COMMENT'),
  y as (
    select *
      from ama3
     where comment = 'COMMENT')
select generators.id,
       x.METRIC,
       x.is_best,
       y.METRIC,
       y.is_best
  from generators
         join (
           select *, ("metrics.test.elbo" in (select min("metrics.test.elbo")
                                        from x
                                       group by generator)) as is_best
             from x
         ) as x
             on generators.key = x.generator
         cross join (
           select *, ("metrics.test.elbo" in (select min("metrics.test.elbo")
                                        from y
                                       group by generator)) as is_best
             from y
         ) as y
             on generators.key = y.generator
             and x.N = y.N
             and x.beta_z = y.beta_z
             and x.beta_d = y.beta_d
