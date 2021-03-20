with kltune as (
  select *
    from ama4
   where comment in ('kltune2')
)
select generators.id,
       kltune."metrics.test.elbo",
       kltune."z0z3.compiled_action.train",
       (kltune."metrics.test.elbo" in
       (select min("metrics.test.elbo")
          from kltune
         group by generator)) as is_best
  from generators
         join kltune
             on generators.key = kltune.generator

